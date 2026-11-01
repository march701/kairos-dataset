"""MiniLM worker: keep PyTorch out of the LightGBM process."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'

import torch
import numpy as np

ENCODER_REPO = 'sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
ENCODER_FILES = ['config.json', 'model.safetensors', 'tokenizer.json',
                 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt']
MAX_SEQUENCE_LENGTH = 256
CONTENT_TOKENS = MAX_SEQUENCE_LENGTH - 2


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def digest_texts(texts) -> str:
    value = hashlib.sha256()
    for text in texts:
        value.update((text if isinstance(text, str) else '').encode('utf-8'))
        value.update(b'\n')
    return value.hexdigest()


def ensure_encoder(model_dir: Path) -> Path:
    from huggingface_hub import snapshot_download
    model_dir.mkdir(parents=True, exist_ok=True)
    if not all((model_dir / name).is_file() for name in ENCODER_FILES):
        snapshot_download(ENCODER_REPO, revision=ENCODER_REVISION, local_dir=model_dir,
                         allow_patterns=ENCODER_FILES, max_workers=2)
    missing = [name for name in ENCODER_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'MiniLM files are missing: {missing}')
    write_json(model_dir / 'encoder_info.json',
               {'repository': ENCODER_REPO, 'revision': ENCODER_REVISION, 'files': ENCODER_FILES})
    return model_dir


def encode_full_texts(texts, destination: Path, model_dir: Path, *, batch_texts: int = 8,
                      batch_chunks: int = 16) -> dict:
    """Encode every text in 254-token chunks and pool to one normalized vector."""
    import transformers
    from transformers import AutoModel, AutoTokenizer

    destination.parent.mkdir(parents=True, exist_ok=True)
    model_dir = ensure_encoder(model_dir)
    signature = {
        'rows': len(texts), 'text_sha256': digest_texts(texts),
        'model_revision': ENCODER_REVISION, 'sequence_length': MAX_SEQUENCE_LENGTH,
        'content_tokens_per_chunk': CONTENT_TOKENS,
        'pooling': 'token-weighted attention mean of chunks; L2-normalized post vector',
    }
    progress_path = destination.with_suffix('.progress.json')
    completed, chunks_done = 0, 0
    if progress_path.exists() and destination.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get('signature') != signature:
            raise ValueError('Embedding cache signature differs from current texts/model')
        completed = int(progress['completed_rows'])
        chunks_done = int(progress.get('chunks_done', 0))
        if completed == len(texts):
            return progress

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                        use_safetensors=True).eval().to('cpu')
    matrix = np.lib.format.open_memmap(destination,
        mode='r+' if completed and destination.exists() else 'w+', dtype=np.float32,
        shape=(len(texts), 384))
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(completed, len(texts), batch_texts):
            batch = texts[offset:offset + batch_texts]
            chunk_ids, owners, weights = [], [], []
            for local_index, text in enumerate(batch):
                tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text if isinstance(text, str) else ''))
                pieces = [tokens[i:i + CONTENT_TOKENS] for i in range(0, len(tokens), CONTENT_TOKENS)] or [[]]
                for piece in pieces:
                    prepared = tokenizer.prepare_for_model(piece, add_special_tokens=True,
                        truncation=True, max_length=MAX_SEQUENCE_LENGTH, return_attention_mask=True)
                    chunk_ids.append(prepared['input_ids'])
                    owners.append(local_index)
                    weights.append(max(len(piece), 1))
            chunks_done += len(chunk_ids)
            sums = np.zeros((len(batch), 384), dtype=np.float32)
            totals = np.zeros(len(batch), dtype=np.float32)
            owners = np.asarray(owners, dtype=np.int32)
            weights = np.asarray(weights, dtype=np.float32)
            for start in range(0, len(chunk_ids), batch_chunks):
                stop = min(start + batch_chunks, len(chunk_ids))
                encoded = tokenizer.pad([{'input_ids': ids} for ids in chunk_ids[start:stop]],
                                        padding=True, return_tensors='pt')
                hidden = encoder(**encoded).last_hidden_state
                mask = encoded['attention_mask'].unsqueeze(-1)
                vectors = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
                local_owners, local_weights = owners[start:stop], weights[start:stop]
                np.add.at(sums, local_owners, vectors * local_weights[:, None])
                np.add.at(totals, local_owners, local_weights)
            vectors = sums / np.maximum(totals[:, None], 1.0)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            if not np.isfinite(vectors).all():
                raise ValueError('MiniLM produced non-finite embeddings')
            end = offset + len(batch)
            matrix[offset:end] = vectors
            if end % 128 == 0 or end == len(texts):
                matrix.flush()
                progress = {
                    'signature': signature, 'completed_rows': end, 'chunks_done': chunks_done,
                    'elapsed_seconds': time.monotonic() - started,
                    'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                }
                write_json(progress_path, progress)
                print(f'Encoded {end}/{len(texts)} posts', flush=True)
    matrix.flush()
    del encoder, matrix
    gc.collect()
    return json.loads(progress_path.read_text())


if __name__ == '__main__':
    texts_path, destination, model_dir = map(Path, sys.argv[1:])
    texts = json.loads(texts_path.read_text(encoding='utf-8'))
    encode_full_texts(texts, destination, model_dir)
