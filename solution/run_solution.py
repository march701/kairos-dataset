"""Train or run the single full-text MiniLM + LightGBM solution."""
from __future__ import annotations

import argparse
from collections import deque
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from extra_features_4 import EXTRA_COLUMNS, build_extra_features
from kairos import features
from kairos.data import load_followers, load_labels, load_posts
from kairos.label import author_level

SEED = 42
KEYS = ['author', 'post']
NUMERIC = features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM + EXTRA_COLUMNS
TEXT_FIELDS = ['chars', 'words', 'lines', 'paragraphs', 'hashtags', 'links',
               'mentions', 'questions', 'exclamations']
RELATIVE_COLUMNS = [f'{field}_{comparison}' for field in TEXT_FIELDS
                    for comparison in ('vs_history', 'vs_recent20', 'z_history')]
RELATIVE_COLUMNS += ['prior_posts_log', 'content_type_share20', 'gap_vs_history',
                     'jaccard_previous', 'jaccard_last5', 'jaccard_last20']
MAX_SEQUENCE_LENGTH = 256

def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def metrics(y_true, predicted) -> dict:
    return {
        'macro_f1': float(f1_score(y_true, predicted, average='macro', labels=[0, 1, 2])),
        'qwk': float(cohen_kappa_score(y_true, predicted, weights='quadratic')),
        'accuracy': float(accuracy_score(y_true, predicted)),
        'per_class_f1': f1_score(y_true, predicted, average=None, labels=[0, 1, 2]).tolist(),
        'confusion_matrix': confusion_matrix(y_true, predicted, labels=[0, 1, 2]).tolist(),
    }


def build_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build labelled-post features from the local release; media is not accessed."""
    posts = load_posts(data_dir, own_only=True)
    labels = load_labels(data_dir)
    levelled = author_level(posts)
    base = features.build(levelled, load_followers(data_dir))
    extra = build_extra_features(posts)
    table = labels.merge(base, on=KEYS, how='left', validate='one_to_one')
    table = table.merge(extra, on=KEYS, how='left', validate='one_to_one')
    if table['level'].isna().any():
        raise ValueError('Some labelled posts lack Improved 4 tabular features')
    table['text'] = table['text'].fillna('')
    table['content_type'] = table['content_type'].fillna('unknown')
    table[EXTRA_COLUMNS] = table[EXTRA_COLUMNS].replace([np.inf, -np.inf], 0).fillna(0)
    forbidden = {'reactions', 'log_reactions', 'resid', 'label', 'class', 'split', 'author', 'post'}
    if forbidden.intersection(NUMERIC):
        raise ValueError('Target fields or identifiers cannot be model inputs')
    keep = [*KEYS, 'split', 'label', 'text', 'content_type', *NUMERIC]
    return table[keep].copy(), posts


def _relative_text_values(text: str) -> list[float]:
    paragraphs = [part for part in re.split(r'\n\s*\n', text) if part.strip()]
    values = [len(text), len(text.split()), len(text.splitlines()), len(paragraphs),
              len(re.findall(r'#(?:\w+|⟨[^⟩]*⟩)', text)),
              len(re.findall(r'https?://|⟨URL⟩', text)),
              len(re.findall(r'@\w+|⟨HANDLE⟩|⟨(?:PERSON|AUTHOR)[^⟩]*⟩', text)),
              text.count('?'), text.count('!')]
    return [math.log1p(value) for value in values]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def author_history_features(posts: pd.DataFrame, labelled_keys) -> dict:
    """Compute author-relative text features from earlier posts only."""
    wanted = set(labelled_keys)
    result = {}
    for author, group in posts.groupby('author', sort=False):
        group = group.dropna(subset=['posted_at']).sort_values(['posted_at', 'post'], kind='mergesort')
        recent = deque(maxlen=20)
        count = 0
        totals = [0.0] * len(TEXT_FIELDS)
        squares = [0.0] * len(TEXT_FIELDS)
        previous_time = None
        gap_total = 0.0
        gap_count = 0
        records = group.to_dict('records')
        cursor = 0
        while cursor < len(records):
            timestamp = records[cursor]['posted_at']
            same_time = []
            while cursor < len(records) and records[cursor]['posted_at'] == timestamp:
                same_time.append(records[cursor])
                cursor += 1
            gap = (math.log1p((timestamp - previous_time).total_seconds() / 86400)
                   if previous_time is not None else 0.0)
            pending = []
            for post in same_time:
                text = post.get('text') or ''
                values = _relative_text_values(text)
                words = set(re.findall(r'\w+', text.lower()))
                kind = post.get('content_type') or 'unknown'
                row = []
                for index, value in enumerate(values):
                    mean = totals[index] / count if count else value
                    recent_mean = (sum(item[0][index] for item in recent) / len(recent)
                                   if recent else value)
                    variance = max(0.0, squares[index] / count - mean * mean) if count else 0.0
                    z_score = (value - mean) / math.sqrt(variance) if variance > 1e-8 else 0.0
                    row.extend([value - mean, value - recent_mean,
                                max(-10.0, min(10.0, z_score))])
                similarities = [_jaccard(words, item[1]) for item in recent]
                row.extend([
                    math.log1p(count),
                    sum(item[2] == kind for item in recent) / len(recent) if recent else 0.0,
                    gap - gap_total / gap_count if gap_count else 0.0,
                    similarities[-1] if similarities else 0.0,
                    sum(similarities[-5:]) / len(similarities[-5:]) if similarities else 0.0,
                    sum(similarities) / len(similarities) if similarities else 0.0,
                ])
                key = (author, post['post'])
                if key in wanted:
                    result[key] = row
                pending.append((values, words, kind))
            for values, words, kind in pending:
                count += 1
                for index, value in enumerate(values):
                    totals[index] += value
                    squares[index] += value * value
                recent.append((values, words, kind))
            if previous_time is not None:
                gap_total += gap
                gap_count += 1
            previous_time = timestamp
    if set(result) != wanted:
        raise ValueError(f'Author history features missing {len(wanted - set(result))} labelled posts')
    return result


def encode_full_texts(texts, destination: Path, model_dir: Path) -> dict:
    """Run MiniLM in a fresh interpreter, isolated from LightGBM's OpenMP."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='kairos-encoder-') as temporary:
        texts_path = Path(temporary) / 'texts.json'
        texts_path.write_text(json.dumps(texts, ensure_ascii=False), encoding='utf-8')
        subprocess.run([sys.executable, '-X', 'faulthandler',
                        str(Path(__file__).with_name('encoder_worker.py')),
                        str(texts_path), str(destination.resolve()), str(model_dir.resolve())],
                       check=True)
    return json.loads(destination.with_suffix('.progress.json').read_text())


def fit_tabular(train: pd.DataFrame, evaluation: pd.DataFrame):
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    fit_num = scaler.fit_transform(train[NUMERIC]).astype(np.float32)
    eval_num = scaler.transform(evaluation[NUMERIC]).astype(np.float32)
    fit_cat = encoder.fit_transform(train[['content_type']])
    eval_cat = encoder.transform(evaluation[['content_type']])
    return (np.column_stack([fit_num, fit_cat]).astype(np.float32),
            np.column_stack([eval_num, eval_cat]).astype(np.float32), scaler, encoder)


def classifier(labels):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3)
    if (counts == 0).any():
        raise ValueError('All three classes must be present in official training data')
    weights = {label: float(math.sqrt(len(labels) / (3 * count)))
               for label, count in enumerate(counts)}
    return lgb.LGBMClassifier(
        objective='multiclass', n_estimators=300, learning_rate=.05, num_leaves=31,
        min_child_samples=40, reg_lambda=10., class_weight=weights, n_jobs=2,
        random_state=SEED, verbosity=-1, deterministic=True, force_col_wise=True, max_bin=127,
    )


def get_table(data_dir: Path, output: Path, *, force_features: bool = False):
    cache = output / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    table_path, relative_path = cache / 'dataset.pkl', cache / 'author_relative.npy'
    if table_path.exists() and relative_path.exists() and not force_features:
        table = pd.read_pickle(table_path)
        relative = np.load(relative_path, mmap_mode='r', allow_pickle=False)
        if relative.shape != (len(table), len(RELATIVE_COLUMNS)):
            raise ValueError('Author-history cache shape is invalid; rerun with --force-features')
        return table, relative
    print('Building tabular features and author history from local data...', flush=True)
    table, posts = build_dataset(data_dir)
    keys = [tuple(row) for row in table[KEYS].itertuples(index=False, name=None)]
    history = author_history_features(posts, keys)
    matrix = np.asarray([history[key] for key in keys], dtype=np.float32)
    if matrix.shape != (len(table), len(RELATIVE_COLUMNS)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid author-history feature matrix')
    table.to_pickle(table_path)
    np.save(relative_path, matrix)
    del posts, history, matrix
    gc.collect()
    return table, np.load(relative_path, mmap_mode='r', allow_pickle=False)


def train(data_dir: Path, output: Path, *, force_features: bool = False, force: bool = False):
    if (output / 'training_report.json').exists() and not force:
        raise SystemExit('A completed training run exists; choose another --output or pass --force.')
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    table, relative = get_table(data_dir, output, force_features=force_features)
    keys = [tuple(row) for row in table[KEYS].itertuples(index=False, name=None)]
    position = {key: index for index, key in enumerate(keys)}
    train = table.loc[table['split'].eq('train')].reset_index(drop=True)
    test = table.loc[table['split'].eq('test')].reset_index(drop=True)
    train_rows = np.asarray([position[tuple(row)] for row in train[KEYS].itertuples(index=False, name=None)])
    test_rows = np.asarray([position[tuple(row)] for row in test[KEYS].itertuples(index=False, name=None)])
    if set(train.author) & set(test.author):
        raise ValueError('Official train/test authors overlap')

    embedding_path = output / 'cache' / 'fulltext_embeddings.npy'
    encoder_dir = output / 'encoder'
    texts = table['text'].fillna('').astype(str).tolist()
    encoding = encode_full_texts(texts, embedding_path, encoder_dir)
    embeddings = np.load(embedding_path, mmap_mode='r', allow_pickle=False)
    if embeddings.shape != (len(table), 384):
        raise ValueError('Unexpected full-text embedding matrix shape')

    train_tab, test_tab, scaler, category_encoder = fit_tabular(train, test)
    x_train = np.column_stack([train_tab, embeddings[train_rows], relative[train_rows]]).astype(np.float32, copy=False)
    x_test = np.column_stack([test_tab, embeddings[test_rows], relative[test_rows]]).astype(np.float32, copy=False)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError('Non-finite model features')
    model = classifier(train['label'].to_numpy(dtype=np.int64))
    fit_started = time.monotonic()
    with threadpool_limits(limits=2):
        model.fit(x_train, train['label'].to_numpy(dtype=np.int64))
    training_seconds = time.monotonic() - fit_started
    probabilities = model.predict_proba(x_test).astype(np.float32)
    predicted = probabilities.argmax(axis=1)
    result = metrics(test['label'].to_numpy(dtype=np.int64), predicted)

    test[KEYS].assign(label=predicted).to_csv(output / 'test_predictions.csv', index=False)
    joblib.dump(dict(scaler=scaler, encoder=category_encoder, numeric_columns=NUMERIC,
                     relative_columns=RELATIVE_COLUMNS, embedding_dimensions=384),
                output / 'preprocessing.joblib')
    model.booster_.save_model(str(output / 'lightgbm_model.txt'))
    np.save(output / 'test_probabilities.npy', probabilities)
    report = {
        'model': 'single LightGBM: full-text chunked MiniLM embeddings + tabular + author-relative text features',
        'split': 'fit all official train; evaluate once on official test',
        'train_rows': len(train), 'test_rows': len(test),
        'validation_macro_f1_reference': 0.6405600081730775,
        'embedding_strategy': 'token-weighted average of attention-masked mean vectors over non-overlapping 254-token chunks; L2 normalized',
        'embedding_sequence_length': MAX_SEQUENCE_LENGTH,
        'relative_feature_count': len(RELATIVE_COLUMNS), 'ensemble': False,
        'parameters': model.get_params(), 'metrics': result,
        'training_seconds': training_seconds, 'elapsed_seconds': time.monotonic() - started,
        'embedding_progress': encoding,
    }
    write_json(output / 'training_report.json', report)
    print(json.dumps(result, indent=2), flush=True)


def predict(output: Path):
    table_path = output / 'cache' / 'dataset.pkl'
    relative_path = output / 'cache' / 'author_relative.npy'
    embedding_path = output / 'cache' / 'fulltext_embeddings.npy'
    required = [table_path, relative_path, embedding_path, output / 'lightgbm_model.txt',
                output / 'preprocessing.joblib']
    if not all(path.is_file() for path in required):
        raise SystemExit('Trained model/cache not found. Run the train command first.')
    table = pd.read_pickle(table_path)
    relative = np.load(relative_path, mmap_mode='r', allow_pickle=False)
    embeddings = np.load(embedding_path, mmap_mode='r', allow_pickle=False)
    test = table.loc[table['split'].eq('test')].reset_index(drop=True)
    keys = [tuple(row) for row in table[KEYS].itertuples(index=False, name=None)]
    position = {key: index for index, key in enumerate(keys)}
    rows = np.asarray([position[tuple(row)] for row in test[KEYS].itertuples(index=False, name=None)])
    prep = joblib.load(output / 'preprocessing.joblib')
    numeric = prep['scaler'].transform(test[NUMERIC]).astype(np.float32)
    categorical = prep['encoder'].transform(test[['content_type']])
    tabular = np.column_stack([numeric, categorical]).astype(np.float32, copy=False)
    matrix = np.column_stack([tabular, embeddings[rows], relative[rows]]).astype(np.float32, copy=False)
    model = lgb.Booster(model_file=str(output / 'lightgbm_model.txt'))
    probabilities = model.predict(matrix).astype(np.float32)
    predicted = probabilities.argmax(axis=1)
    result = metrics(test['label'].to_numpy(dtype=np.int64), predicted)
    test[KEYS].assign(label=predicted).to_csv(output / 'test_predictions.csv', index=False)
    np.save(output / 'test_probabilities.npy', probabilities)
    write_json(output / 'prediction_metrics.json', {'split': 'official test', 'metrics': result,
                                                    'rows': len(test), 'ensemble': False})
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--data', type=Path, default=PROJECT / 'data')
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/solution')
    parser.add_argument('--force-features', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    data_dir, output = args.data.resolve(), args.output.resolve()
    if args.command == 'train':
        train(data_dir, output, force_features=args.force_features, force=args.force)
    else:
        predict(output)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
