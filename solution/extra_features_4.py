
"""Additional features for Improved 4."""

from __future__ import annotations

import re
from collections import deque

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


EXTRA_COLUMNS = [
    "avg_sentence_chars",
    "first_paragraph_chars",
    "n_paragraphs",
    "n_exclamations",
    "n_question_marks",
    "n_emojis",
    "uppercase_ratio",
    "avg_word_length",
    "first_line_chars",
    "first_line_question",
    "first_line_number",
    "has_cta",
    "is_weekend",
    "time_morning",
    "time_afternoon",
    "time_evening",
    "time_night",
    "posts_last_7d",
    "posts_last_30d",
    "hist_mean_log",
    "hist_std_log",
    "recent_mean_5",
    "recent_mean_20",
    "recent_std_10",
    "recent_trend",
    "topic_similarity_prev",
    "topic_similarity_last5",
]


def text_features(text: str) -> dict:
    text = text if isinstance(text, str) else ""

    words = re.findall(r"\b\w+\b", text)
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", text)
        if s.strip()
    ]
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    first_line = lines[0] if lines else ""
    first_paragraph = paragraphs[0] if paragraphs else ""

    letters = [c for c in text if c.isalpha()]

    cta_pattern = (
        r"\b(comment|share|follow|subscribe|repost|"
        r"agree|thoughts|let me know)\b"
    )

    emoji_count = sum(
        0x1F300 <= ord(c) <= 0x1FAFF
        for c in text
    )

    return {
        "avg_sentence_chars": (
            np.mean([len(s) for s in sentences])
            if sentences else 0.0
        ),
        "first_paragraph_chars": len(first_paragraph),
        "n_paragraphs": len(paragraphs),
        "n_exclamations": text.count("!"),
        "n_question_marks": text.count("?"),
        "n_emojis": emoji_count,
        "uppercase_ratio": (
            sum(c.isupper() for c in letters) / len(letters)
            if letters else 0.0
        ),
        "avg_word_length": (
            np.mean([len(w) for w in words])
            if words else 0.0
        ),
        "first_line_chars": len(first_line),
        "first_line_question": int("?" in first_line),
        "first_line_number": int(
            bool(re.search(r"\d", first_line))
        ),
        "has_cta": int(
            bool(re.search(cta_pattern, text, re.I))
        ),
    }


def build_extra_features(posts: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for all posts."""

    df = posts.copy()

    df["text"] = df["text"].fillna("")
    df["posted_at"] = pd.to_datetime(
        df["posted_at"], utc=True
    )

    # Deterministic chronological order.
    df = df.sort_values(
        ["author", "posted_at", "post"]
    ).reset_index(drop=True)

    print("Creating text features...", flush=True)

    text_df = pd.DataFrame(
        [text_features(t) for t in df["text"]]
    )

    df = pd.concat([df, text_df], axis=1)

    print("Creating time features...", flush=True)

    hour = df["posted_at"].dt.hour

    df["is_weekend"] = (
        df["posted_at"].dt.weekday >= 5
    ).astype(int)

    df["time_morning"] = hour.between(6, 11).astype(int)
    df["time_afternoon"] = hour.between(12, 17).astype(int)
    df["time_evening"] = hour.between(18, 23).astype(int)
    df["time_night"] = hour.between(0, 5).astype(int)

    print("Creating author history features...", flush=True)

    df["log_reactions"] = np.log1p(
        df["reactions"].clip(lower=0)
    )

    history_columns = [
        "posts_last_7d",
        "posts_last_30d",
        "hist_mean_log",
        "hist_std_log",
        "recent_mean_5",
        "recent_mean_20",
        "recent_std_10",
        "recent_trend",
    ]

    for col in history_columns:
        df[col] = 0.0

    for _, group in df.groupby("author", sort=False):
        idx = group.index.to_numpy()
        dates = group["posted_at"].to_numpy()
        values = group["log_reactions"].to_numpy(dtype=float)

        # Shift ensures current post reactions are excluded.
        previous = pd.Series(values).shift(1)

        df.loc[idx, "hist_mean_log"] = (
            previous.expanding(min_periods=1)
            .mean().fillna(0).to_numpy()
        )

        df.loc[idx, "hist_std_log"] = (
            previous.expanding(min_periods=2)
            .std().fillna(0).to_numpy()
        )

        df.loc[idx, "recent_mean_5"] = (
            previous.rolling(5, min_periods=1)
            .mean().fillna(0).to_numpy()
        )

        df.loc[idx, "recent_mean_20"] = (
            previous.rolling(20, min_periods=1)
            .mean().fillna(0).to_numpy()
        )

        df.loc[idx, "recent_std_10"] = (
            previous.rolling(10, min_periods=2)
            .std().fillna(0).to_numpy()
        )

        recent5 = df.loc[idx, "recent_mean_5"].to_numpy()
        recent20 = df.loc[idx, "recent_mean_20"].to_numpy()

        df.loc[idx, "recent_trend"] = recent5 - recent20

        for i in range(len(idx)):
            previous_dates = dates[:i]

            df.at[idx[i], "posts_last_7d"] = np.sum(
                (previous_dates >= dates[i] - np.timedelta64(7, "D"))
                & (previous_dates < dates[i])
            )

            df.at[idx[i], "posts_last_30d"] = np.sum(
                (previous_dates >= dates[i] - np.timedelta64(30, "D"))
                & (previous_dates < dates[i])
            )

    print("Creating topic similarity features...", flush=True)

    vectorizer = HashingVectorizer(
        n_features=4096,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        dtype=np.float32,
    )

    X_text = vectorizer.transform(df["text"])

    df["topic_similarity_prev"] = 0.0
    df["topic_similarity_last5"] = 0.0

    for _, group in df.groupby("author", sort=False):
        idx = group.index.to_numpy()

        for i in range(1, len(idx)):
            current = X_text[idx[i]]

            prev = X_text[idx[i - 1]]
            similarity = current.multiply(prev).sum()

            df.at[idx[i], "topic_similarity_prev"] = float(
                similarity
            )

            start = max(0, i - 5)
            similarities = []

            for j in range(start, i):
                old = X_text[idx[j]]
                similarities.append(
                    float(current.multiply(old).sum())
                )

            df.at[idx[i], "topic_similarity_last5"] = (
                float(np.mean(similarities))
                if similarities else 0.0
            )

    return df[["author", "post"] + EXTRA_COLUMNS].copy()
