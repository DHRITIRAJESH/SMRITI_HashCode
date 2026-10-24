#!/usr/bin/env python3
"""
SMRITI full-KV context-length sweep.

Purpose:
  Measure exact-value recall as distractor/filler context grows, while keeping
  the target fact at the beginning of the user message and the query at the end.

Important interpretation:
  This design tests long-context interference/recency under a full cache.
  It does NOT test KV-cache eviction/compression. The target's absolute position
  is fixed; its distance from the final question grows with filler length.

Example:
  python -u src/evaluate_context_sweep.py \
    --data data/benchmark.json \
    --out results/context_sweep_v1.json \
    --lengths 128,512,1024,2048 \
    2>&1 | tee results/context_sweep_v1.log
"""

import argparse
import json
import random
import re
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# Semantically bland filler: no numbers, dates, IDs, rates, or amounts.
FILLER_SENTENCES = [
    "This paragraph describes ordinary office routines and general document handling.",
    "People may review a page, organize folders, and check that the layout is readable.",
    "A clear heading can help readers navigate a longer document.",
    "Some notes describe routine planning, coordination, and the order of simple tasks.",
    "A document can contain sections, short explanations, and neutral background material.",
    "Readers sometimes pause between sections before continuing to the next paragraph.",
    "Consistent formatting can make a report easier to scan and understand.",
    "The following material is general background text and contains no task-specific records.",
]


def parse_args():
    p = argparse.ArgumentParser(description="SMRITI full-cache context length sweep")
    p.add_argument("--data", default="data/benchmark.json")
    p.add_argument("--out", default="results/context_sweep_v1.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lengths", default="128,512,1024,2048",
                   help="Approximate filler-token targets, comma separated")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--limit", type=int, default=0,
                   help="Optional number of examples for a smoke test; 0=all")
    p.add_argument("--resume", action="store_true",
                   help="Skip (case_id, requested_filler_tokens) already present in output")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def load_records(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Accept common wrapper formats.
        for key in ("examples", "records", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must be a list or contain an examples/records/data list.")
    return data


def get_field(row, candidates, required=True):
    for key in candidates:
        val = row.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    if required:
        raise KeyError(f"Missing required field; tried {candidates}. Row keys: {list(row)}")
    return ""


def build_filler(tokenizer, desired_tokens, rng):
    """Create neutral filler and trim its token IDs to requested approximate size."""
    pieces = []
    while len(pieces) < desired_tokens + 100:
        sentence = rng.choice(FILLER_SENTENCES)
        pieces.append(sentence)
    text = " ".join(pieces)
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:desired_tokens]
    # Decode truncated IDs. Final prompt token count is measured separately.
    return tokenizer.decode(ids, skip_special_tokens=True)


def render_messages(tokenizer, fact_text, filler, question):
    user_content = (
        "Remember the following record exactly:\n"
        f"{fact_text}\n\n"
        "Background material:\n"
        f"{filler}\n\n"
        f"Question: {question}\n"
        "Return only the requested exact value. Do not explain or list alternatives."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise recall assistant. Use the supplied record only. "
                "Do not calculate, infer, invent, or list alternatives. "
                "Return only the exact requested value."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return tokenized, messages


def normalize_basic(value):
    # This is intentionally conservative. Reported strict match compares stripped
    # output with stripped expected; a separate recoverability flag is recorded.
    return re.sub(r"\s+", " ", str(value)).strip()


def main():
    args = parse_args()
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    if not lengths or any(x < 0 for x in lengths):
        raise ValueError("--lengths must be comma-separated nonnegative integers")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Loading tokenizer/model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    records = load_records(args.data)
    if args.limit:
        records = records[:args.limit]
    print(f"Benchmark examples: {len(records)}", flush=True)
    print(f"Filler-token targets: {lengths}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    completed = set()

    if args.resume and out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            completed = {
                (r.get("case_id"), r.get("requested_filler_tokens"))
                for r in results
            }
            print(f"Resuming: loaded {len(results)} existing rows", flush=True)
        except Exception as e:
            raise RuntimeError(f"Cannot resume from {out_path}: {e}")

    total_jobs = len(records) * len(lengths)
    done = len(completed)
    # Reuse identical filler text for every example at a given condition so
    # the context condition is paired and does not vary by record.
    filler_by_length = {
        n: build_filler(tokenizer, n, random.Random(args.seed + n))
        for n in lengths
    }

    for row in records:
        case_id = get_field(row, ["case_id", "id"])
        fact_group_id = get_field(row, ["fact_group_id", "group_id"], required=False)
        language = get_field(row, ["language", "lang"], required=False)
        fact_type = get_field(row, ["fact_type", "type"], required=False)
        fact_text = get_field(row, ["fact", "text", "statement"])
        question = get_field(row, ["question"])
        expected = get_field(row, ["expected", "answer", "target"])

        for requested_len in lengths:
            key = (case_id, requested_len)
            if key in completed:
                continue

            filler = filler_by_length[requested_len]
            tokenized, messages = render_messages(tokenizer, fact_text, filler, question)

            # Transformers versions differ: chat_template may return a
            # BatchEncoding/dict or, without return_dict support, a tensor.
            if isinstance(tokenized, dict) or hasattr(tokenized, "keys"):
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized.get("attention_mask")
            else:
                input_ids = tokenized
                attention_mask = None

            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            prompt_tokens = int(input_ids.shape[-1])

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            elif not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

            # Inputs are moved to the model's input embedding device.
            input_device = model.get_input_embeddings().weight.device
            input_ids = input_ids.to(input_device)
            attention_mask = attention_mask.to(input_device)

            t0 = time.time()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,  # full-cache baseline
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[0, input_ids.shape[-1]:]
            prediction = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            elapsed = time.time() - t0

            pred_norm = normalize_basic(prediction)
            exp_norm = normalize_basic(expected)
            # Strict baseline flag: whitespace-normalized string equality only.
            strict_match = pred_norm == exp_norm
            # Recall/recoverability signal only, not exact correctness. Conservative
            # substring check is retained as a diagnostic, not a primary metric.
            recoverable = bool(exp_norm and exp_norm in pred_norm)

            result = {
                "case_id": case_id,
                "fact_group_id": fact_group_id,
                "language": language,
                "fact_type": fact_type,
                "requested_filler_tokens": requested_len,
                "measured_prompt_tokens": prompt_tokens,
                "fact": fact_text,
                "question": question,
                "expected": expected,
                "predicted": prediction,
                "strict_match_whitespace_normalized": strict_match,
                "expected_value_appears": recoverable,
                "elapsed_seconds": round(elapsed, 3),
                "cache_mode": "full",
                "target_position": "beginning_of_user_record",
                "seed": args.seed,
                "model": args.model,
            }
            results.append(result)
            completed.add(key)
            done += 1

            # Incremental checkpoint after every evaluation.
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(out_path)

            print(
                f"[{done}/{total_jobs}] {case_id} filler={requested_len} "
                f"prompt_tokens={prompt_tokens} strict={strict_match} "
                f"recoverable={recoverable} pred={prediction[:100]!r}",
                flush=True,
            )

    print("\n" + "=" * 68)
    print("CONTEXT SWEEP SUMMARY")
    print("=" * 68)
    for requested_len in lengths:
        subset = [r for r in results if r["requested_filler_tokens"] == requested_len]
        strict = sum(r["strict_match_whitespace_normalized"] is True for r in subset)
        recov = sum(r["expected_value_appears"] is True for r in subset)
        n = len(subset)
        print(
            f"filler={requested_len:5d}  n={n:3d}  "
            f"strict={strict}/{n} ({strict/n*100:.2f}%)  "
            f"recoverable={recov}/{n} ({recov/n*100:.2f}%)"
            if n else f"filler={requested_len}: no results"
        )

    print(f"\nSaved: {out_path}", flush=True)
    print("Note: exact equality here is whitespace-normalized only. "
          "Audit answer normalization before making research claims.")


if __name__ == "__main__":
    main()
