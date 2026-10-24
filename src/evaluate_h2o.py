import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BENCHMARK_PATH = "data/benchmark.json"
RESULTS_PATH = "data/h2o_results.json"

KV_BUDGETS = [0.75, 0.50, 0.25]


def normalize(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.replace(",", "")
    return text.strip()


def load_benchmark(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().lstrip()

    if raw.startswith("["):
        return json.loads(raw)

    repaired = "[\n{" + raw
    return json.loads(repaired)


def get_language(item, index):
    language = item.get("language")

    if language:
        return language

    fact = item.get("fact", "")

    if any("\u0900" <= char <= "\u097F" for char in fact):
        return "hi"

    return "unknown"


def check_value_recall(predicted, expected):
    return normalize(expected) in normalize(predicted)


def calculate_token_importance(attentions):
    """
    Calculate one importance score per token.

    attentions:
        list of tensors
        [batch, attention_heads, query_tokens, key_tokens]

    We aggregate across layers, heads and query positions.
    """

    scores = None

    for attention in attentions:

        if attention is None:
            continue

        # Remove batch dimension
        # [heads, queries, keys]
        attention = attention[0]

        # Sum attention received by each key token.
        #
        # Result:
        # [keys]
        layer_scores = attention.sum(dim=(0, 1))

        if scores is None:
            scores = layer_scores
        else:
            scores = scores + layer_scores

    scores = scores.float()

    return scores


def select_h2o_tokens(scores, budget_ratio, recent_ratio=0.20):
    """
    H2O-style selection.

    Keep:
      1. important historical tokens
      2. recent tokens

    The total number of retained tokens is controlled
    by budget_ratio.
    """

    seq_len = scores.shape[0]

    budget = max(
        1,
        int(seq_len * budget_ratio)
    )

    if budget >= seq_len:
        return torch.arange(seq_len, device=scores.device)

    recent_count = min(
        max(1, int(budget * recent_ratio)),
        budget
    )

    recent_start = seq_len - recent_count

    recent_indices = torch.arange(
        recent_start,
        seq_len,
        device=scores.device
    )

    historical_count = budget - recent_count

    if historical_count <= 0:
        return recent_indices

    historical_indices = torch.arange(
        0,
        recent_start,
        device=scores.device
    )

    historical_scores = scores[:recent_start]

    _, top_indices = torch.topk(
        historical_scores,
        k=historical_count
    )

    selected = torch.cat(
        [
            top_indices,
            recent_indices
        ]
    )

    selected, _ = torch.sort(selected)

    return selected


def compress_cache(cache, selected_indices):
    """
    Create a new DynamicCache containing only
    the selected token positions.
    """

    new_cache = DynamicCache()

    for layer_idx, layer in enumerate(cache.layers):

        keys = layer.keys
        values = layer.values

        selected = selected_indices.to(keys.device)

        new_keys = keys.index_select(
            2,
            selected
        )

        new_values = values.index_select(
            2,
            selected
        )

        new_cache.update(
            new_keys,
            new_values,
            layer_idx
        )

    return new_cache


def generate_with_h2o(
    model,
    tokenizer,
    prompt,
    budget_ratio
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    # --------------------------------------------------
    # PREFILL
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=True,
            return_dict=True
        )

    cache = outputs.past_key_values
    attentions = outputs.attentions

    # --------------------------------------------------
    # CALCULATE TOKEN IMPORTANCE
    # --------------------------------------------------

    scores = calculate_token_importance(
        attentions
    )

    # --------------------------------------------------
    # SELECT H2O TOKENS
    # --------------------------------------------------

    selected_indices = select_h2o_tokens(
        scores,
        budget_ratio
    )

    # --------------------------------------------------
    # COMPRESS KV CACHE
    # --------------------------------------------------

    compressed_cache = compress_cache(
        cache,
        selected_indices
    )

    # --------------------------------------------------
    # GENERATE
    # --------------------------------------------------

    last_token = inputs["input_ids"][:, -1:]

    generated_tokens = []

    current_input = last_token

    past_length = compressed_cache.get_seq_length()

    for _ in range(20):

        with torch.no_grad():

            output = model(
                input_ids=current_input,
                past_key_values=compressed_cache,
                use_cache=True,
                return_dict=True
            )

        logits = output.logits[:, -1, :]

        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(token_id)

        compressed_cache = output.past_key_values

        current_input = next_token

    if not generated_tokens:
        return ""

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return generated_text.strip()


def main():

    print("=" * 70)
    print("SMRITI — H2O KV COMPRESSION BENCHMARK")
    print("=" * 70)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    model.eval()

    print("Model loaded.")

    benchmark = load_benchmark(
        BENCHMARK_PATH
    )

    print(
        "\nBenchmark examples:",
        len(benchmark)
    )

    all_results = {}

    for budget_ratio in KV_BUDGETS:

        budget_name = f"{int(budget_ratio * 100)}%"

        print("\n")
        print("=" * 70)
        print(f"H2O KV BUDGET: {budget_name}")
        print("=" * 70)

        results = []

        total = 0
        exact_correct = 0
        recall_correct = 0

        language_stats = {}
        type_stats = {}

        for index, item in enumerate(
            benchmark,
            start=1
        ):

            case_id = item.get(
                "case_id",
                f"case_{index:03d}"
            )

            language = get_language(
                item,
                index
            )

            fact_type = item.get(
                "fact_type",
                "unknown"
            )

            fact = item["fact"]
            question = item["question"]
            expected = item["answer"]

            prompt = f"""You are an exact-value extraction system.

Read the fact and answer the question.

Return ONLY the exact answer value.
Do not explain.
Do not provide steps.
Do not provide alternatives.

Fact:
{fact}

Question:
{question}

Answer:"""

            print(
                f"\n[{index}/{len(benchmark)}] "
                f"{case_id} | {language} | {fact_type}"
            )

            predicted = generate_with_h2o(
                model,
                tokenizer,
                prompt,
                budget_ratio
            )

            exact_match = (
                normalize(predicted)
                == normalize(expected)
            )

            value_recall = check_value_recall(
                predicted,
                expected
            )

            print(
                "Expected:",
                expected
            )

            print(
                "Predicted:",
                predicted
            )

            print(
                "Recall:",
                value_recall
            )

            total += 1

            if exact_match:
                exact_correct += 1

            if value_recall:
                recall_correct += 1

            if language not in language_stats:

                language_stats[language] = {
                    "total": 0,
                    "exact_correct": 0,
                    "recall_correct": 0
                }

            language_stats[language]["total"] += 1

            if exact_match:
                language_stats[language][
                    "exact_correct"
                ] += 1

            if value_recall:
                language_stats[language][
                    "recall_correct"
                ] += 1

            if fact_type not in type_stats:

                type_stats[fact_type] = {
                    "total": 0,
                    "exact_correct": 0,
                    "recall_correct": 0
                }

            type_stats[fact_type]["total"] += 1

            if exact_match:
                type_stats[fact_type][
                    "exact_correct"
                ] += 1

            if value_recall:
                type_stats[fact_type][
                    "recall_correct"
                ] += 1

            results.append({
                "case_id": case_id,
                "language": language,
                "fact_type": fact_type,
                "fact": fact,
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "exact_match": exact_match,
                "value_recall": value_recall
            })

        exact_accuracy = (
            exact_correct / total * 100
        )

        recall_accuracy = (
            recall_correct / total * 100
        )

        print("\n")
        print("-" * 70)
        print(f"H2O {budget_name} RESULTS")
        print("-" * 70)

        print(
            f"Exact match: "
            f"{exact_correct}/{total} "
            f"({exact_accuracy:.2f}%)"
        )

        print(
            f"Value recall: "
            f"{recall_correct}/{total} "
            f"({recall_accuracy:.2f}%)"
        )

        print("\nBy language:")

        for language, stats in language_stats.items():

            t = stats["total"]

            print(
                f"{language:<12}"
                f"Recall: "
                f"{stats['recall_correct']}/{t}"
                f" ({stats['recall_correct']/t*100:.2f}%)"
            )

        print("\nBy fact type:")

        for fact_type, stats in type_stats.items():

            t = stats["total"]

            print(
                f"{fact_type:<15}"
                f"Recall: "
                f"{stats['recall_correct']}/{t}"
                f" ({stats['recall_correct']/t*100:.2f}%)"
            )

        all_results[budget_name] = {
            "kv_budget": budget_ratio,
            "overall": {
                "total": total,
                "exact_match": {
                    "correct": exact_correct,
                    "accuracy": exact_accuracy
                },
                "value_recall": {
                    "correct": recall_correct,
                    "accuracy": recall_accuracy
                }
            },
            "by_language": language_stats,
            "by_fact_type": type_stats,
            "examples": results
        }

    output = {
        "model": MODEL_NAME,
        "method": "H2O-style KV compression",
        "benchmark": BENCHMARK_PATH,
        "results": all_results
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n")
    print("=" * 70)
    print("H2O EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        "Saved results to:",
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()
