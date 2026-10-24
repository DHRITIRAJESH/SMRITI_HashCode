import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BENCHMARK_PATH = "data/benchmark.json"
RESULTS_PATH = "data/baseline_results.json"


def normalize(text):
    """
    Normalize text for comparison.

    We intentionally do NOT remove punctuation such as:
    %, -, :, / because these can be important in IDs,
    dates, rates, etc.
    """
    text = str(text).strip().lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove commas only.
    # This allows:
    # ₹7,43,281
    # ₹743281
    # to be treated as the same numeric value.
    text = text.replace(",", "")

    return text.strip()


def load_benchmark(path):
    """
    Load the benchmark without modifying the actual file.

    The current benchmark file is missing the opening
    '[' and '{' at the beginning, so we repair it
    IN MEMORY only.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().lstrip()

    if raw.startswith("["):
        return json.loads(raw)

    # Current benchmark starts directly with:
    #
    # "fact_type": ...
    #
    # so repair it in memory.
    repaired = "[\n{" + raw
    return json.loads(repaired)


def generate_answer(model, tokenizer, text):
    """
    Generate a short answer.

    We deliberately limit generation because this is
    an exact-value recall benchmark, not a general QA task.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Only decode newly generated tokens.
    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return generated_text.strip()


def check_value_recall(predicted, expected):
    """
    Check whether the expected exact value appears
    anywhere inside the model's answer.

    Example:

    Expected:
        SMR-804271

    Predicted:
        Reference ID SMR-804271 hai.

    -> True
    """

    predicted_normalized = normalize(predicted)
    expected_normalized = normalize(expected)

    return expected_normalized in predicted_normalized


def main():

    print("=" * 70)
    print("SMRITI EXACT-VALUE MEMORY BENCHMARK")
    print("=" * 70)

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    model.eval()

    print("Model loaded.")

    # --------------------------------------------------
    # Load benchmark
    # --------------------------------------------------

    benchmark = load_benchmark(BENCHMARK_PATH)

    print("\nBenchmark examples:", len(benchmark))

    # --------------------------------------------------
    # Result containers
    # --------------------------------------------------

    results = []

    total = 0

    exact_correct = 0
    recall_correct = 0

    # Language statistics
    language_stats = {}

    # Fact type statistics
    type_stats = {}

    # --------------------------------------------------
    # Run benchmark
    # --------------------------------------------------

    for index, item in enumerate(benchmark, start=1):

        case_id = item.get(
            "case_id",
            f"case_{index:03d}"
        )

        language = item.get("language")

        # The first malformed record in the benchmark
        # does not contain a language field.
        if not language:

            fact = item.get("fact", "")

            if any(
                "\u0900" <= char <= "\u097F"
                for char in fact
            ):
                language = "hi"
            else:
                language = "unknown"

        fact_type = item.get(
            "fact_type",
            "unknown"
        )

        fact = item["fact"]
        question = item["question"]
        expected = item["answer"]

        # --------------------------------------------------
        # Construct prompt
        # --------------------------------------------------

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

        print("\n" + "=" * 70)
        print(f"[{index}/{len(benchmark)}]")
        print("Case:", case_id)
        print("Language:", language)
        print("Type:", fact_type)
        print("Fact:", fact)
        print("Question:", question)

        # --------------------------------------------------
        # Generate
        # --------------------------------------------------

        predicted = generate_answer(
            model,
            tokenizer,
            prompt
        )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        exact_match = (
            normalize(predicted)
            == normalize(expected)
        )

        value_recall = check_value_recall(
            predicted,
            expected
        )

        print("Expected:", expected)
        print("Predicted:", predicted)
        print("Exact match:", exact_match)
        print("Value recall:", value_recall)

        # --------------------------------------------------
        # Update totals
        # --------------------------------------------------

        total += 1

        if exact_match:
            exact_correct += 1

        if value_recall:
            recall_correct += 1

        # --------------------------------------------------
        # Language statistics
        # --------------------------------------------------

        if language not in language_stats:
            language_stats[language] = {
                "total": 0,
                "exact_correct": 0,
                "recall_correct": 0
            }

        language_stats[language]["total"] += 1

        if exact_match:
            language_stats[language]["exact_correct"] += 1

        if value_recall:
            language_stats[language]["recall_correct"] += 1

        # --------------------------------------------------
        # Fact type statistics
        # --------------------------------------------------

        if fact_type not in type_stats:
            type_stats[fact_type] = {
                "total": 0,
                "exact_correct": 0,
                "recall_correct": 0
            }

        type_stats[fact_type]["total"] += 1

        if exact_match:
            type_stats[fact_type]["exact_correct"] += 1

        if value_recall:
            type_stats[fact_type]["recall_correct"] += 1

        # --------------------------------------------------
        # Save individual result
        # --------------------------------------------------

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

    # ======================================================
    # FINAL RESULTS
    # ======================================================

    exact_accuracy = (
        exact_correct / total * 100
        if total > 0
        else 0
    )

    recall_accuracy = (
        recall_correct / total * 100
        if total > 0
        else 0
    )

    print("\n")
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print("Total examples:", total)

    print(
        f"Strict exact match: "
        f"{exact_correct}/{total} "
        f"({exact_accuracy:.2f}%)"
    )

    print(
        f"Exact-value recall: "
        f"{recall_correct}/{total} "
        f"({recall_accuracy:.2f}%)"
    )

    # ======================================================
    # RESULTS BY LANGUAGE
    # ======================================================

    print("\nRESULTS BY LANGUAGE")
    print("-" * 70)

    for language, stats in language_stats.items():

        total_lang = stats["total"]

        exact_acc = (
            stats["exact_correct"]
            / total_lang
            * 100
        )

        recall_acc = (
            stats["recall_correct"]
            / total_lang
            * 100
        )

        print(
            f"{language:<12}"
            f"Exact: {stats['exact_correct']}/{total_lang}"
            f" ({exact_acc:.2f}%)"
            f" | Recall: {stats['recall_correct']}/{total_lang}"
            f" ({recall_acc:.2f}%)"
        )

    # ======================================================
    # RESULTS BY FACT TYPE
    # ======================================================

    print("\nRESULTS BY FACT TYPE")
    print("-" * 70)

    for fact_type, stats in type_stats.items():

        total_type = stats["total"]

        exact_acc = (
            stats["exact_correct"]
            / total_type
            * 100
        )

        recall_acc = (
            stats["recall_correct"]
            / total_type
            * 100
        )

        print(
            f"{fact_type:<15}"
            f"Exact: {stats['exact_correct']}/{total_type}"
            f" ({exact_acc:.2f}%)"
            f" | Recall: {stats['recall_correct']}/{total_type}"
            f" ({recall_acc:.2f}%)"
        )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    output = {
        "model": MODEL_NAME,
        "benchmark": BENCHMARK_PATH,

        "overall": {
            "total": total,

            "strict_exact_match": {
                "correct": exact_correct,
                "accuracy": exact_accuracy
            },

            "exact_value_recall": {
                "correct": recall_correct,
                "accuracy": recall_accuracy
            }
        },

        "by_language": language_stats,

        "by_fact_type": type_stats,

        "examples": results
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

    print("\nSaved results to:")
    print(RESULTS_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
