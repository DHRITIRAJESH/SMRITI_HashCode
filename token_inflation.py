
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CONFIG = "hin_Deva-eng_Latn"

OUT = Path("results/token_inflation")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading FLORES...")
ds = load_dataset("facebook/flores", CONFIG)

def get_stats(text):
    # No chat template or special tokens.
    # Measure the sentence's ordinary tokenization.
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    return {
        "tokens": len(ids),
        "words": len(text.split()),
        "utf8_bytes": len(text.encode("utf-8")),
        "chars": len(text),
    }

def analyze(split_name, split):
    rows = []

    for i, ex in enumerate(split):
        en = ex["sentence_eng_Latn"]
        hi = ex["sentence_hin_Deva"]

        en_s = get_stats(en)
        hi_s = get_stats(hi)

        rows.append({
            "id": i,
            "en_text": en,
            "hi_text": hi,

            "en_tokens": en_s["tokens"],
            "hi_tokens": hi_s["tokens"],

            "en_words": en_s["words"],
            "hi_words": hi_s["words"],

            "en_bytes": en_s["utf8_bytes"],
            "hi_bytes": hi_s["utf8_bytes"],

            "en_chars": en_s["chars"],
            "hi_chars": hi_s["chars"],

            "token_ratio_hi_en": (
                hi_s["tokens"] / en_s["tokens"]
                if en_s["tokens"] else np.nan
            ),

            "tokens_per_en_word": (
                en_s["tokens"] / en_s["words"]
                if en_s["words"] else np.nan
            ),

            "tokens_per_hi_word": (
                hi_s["tokens"] / hi_s["words"]
                if hi_s["words"] else np.nan
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        OUT / f"{split_name}_pairs.csv",
        index=False,
    )

    # Aggregate ratio of total tokens, not average
    # of sentence-level ratios.
    en_total = df["en_tokens"].sum()
    hi_total = df["hi_tokens"].sum()

    ratio = hi_total / en_total if en_total else np.nan

    summary = {
        "split": split_name,
        "n": len(df),
        "en_total_tokens": int(en_total),
        "hi_total_tokens": int(hi_total),
        "aggregate_token_ratio_hi_en": float(ratio),
        "token_inflation_percent": float((ratio - 1) * 100),
        "en_mean_tokens": float(df["en_tokens"].mean()),
        "hi_mean_tokens": float(df["hi_tokens"].mean()),
        "en_median_tokens": float(df["en_tokens"].median()),
        "hi_median_tokens": float(df["hi_tokens"].median()),
        "en_tokens_per_word": float(
            en_total / df["en_words"].sum()
        ),
        "hi_tokens_per_word": float(
            hi_total / df["hi_words"].sum()
        ),
        "en_tokens_per_byte": float(
            en_total / df["en_bytes"].sum()
        ),
        "hi_tokens_per_byte": float(
            hi_total / df["hi_bytes"].sum()
        ),
        "en_mean_bytes": float(df["en_bytes"].mean()),
        "hi_mean_bytes": float(df["hi_bytes"].mean()),
    }

    # Paired bootstrap CI for aggregate token ratio.
    # Resample parallel sentence pairs together.
    rng = np.random.default_rng(42)
    en = df["en_tokens"].to_numpy()
    hi = df["hi_tokens"].to_numpy()

    bootstrap_ratios = []
    for _ in range(5000):
        idx = rng.integers(0, len(df), size=len(df))
        denom = en[idx].sum()
        if denom:
            bootstrap_ratios.append(hi[idx].sum() / denom)

    lo, median, high = np.quantile(
        bootstrap_ratios, [0.025, 0.5, 0.975]
    )

    summary["bootstrap_ratio_ci_95"] = [
        float(lo), float(high)
    ]

    # Descriptive sentence-level ratio stats
    summary["mean_sentence_token_ratio"] = float(
        df["token_ratio_hi_en"].mean()
    )
    summary["median_sentence_token_ratio"] = float(
        df["token_ratio_hi_en"].median()
    )

    return df, summary

summaries = []

for name in ["dev", "devtest"]:
    print(f"\nAnalyzing {name}...")
    _, summary = analyze(name, ds[name])
    summaries.append(summary)
    print(json.dumps(summary, indent=2))

pd.DataFrame(summaries).to_csv(
    OUT / "summary.csv",
    index=False,
)

with open(OUT / "summary.json", "w") as f:
    json.dump(summaries, f, indent=2)

print("\nSaved results to:", OUT.resolve())
