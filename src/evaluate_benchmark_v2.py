import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


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

with open(
    "data/benchmark.json",
    "r",
    encoding="utf-8"
) as f:

    benchmark = json.load(f)


print()
print("Benchmark examples:", len(benchmark))


# --------------------------------------------------
# Ask model
# --------------------------------------------------

def ask_model(fact):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise memory recall assistant. "
                "Answer using only the information in the user's "
                "message. Return only the requested value. "
                "Do not explain, list alternatives, or guess."
            )
        },
        {
            "role": "user",
            "content": (
                f"Remember this information:\n\n"
                f"{fact['fact']}\n\n"
                f"Question: {fact['question']}\n\n"
                "Return only the exact value."
            )
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return answer.strip()



# --------------------------------------------------
# Normalization
# --------------------------------------------------

def normalize(text):

    return (
        text
        .strip()
        .lower()
        .replace(",", "")
    )


# --------------------------------------------------
# Run benchmark
# --------------------------------------------------

results = []


for i, item in enumerate(benchmark):

    print()
    print("=" * 70)

    print(
        f"[{i + 1}/{len(benchmark)}]"
    )

    print(
        "Case:",
        item.get("case_id")
    )

    print(
        "Language:",
        item.get("language")
    )

    print(
        "Type:",
        item.get("fact_type")
    )

    print(
        "Fact:",
        item["fact"]
    )

    print(
        "Question:",
        item["question"]
    )


    predicted = ask_model(item)

    expected = item["answer"]


    correct = (
        normalize(expected)
        == normalize(predicted)
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
        "Exact match:",
        correct
    )


    results.append({

        "case_id": item.get("case_id"),

        "fact_group_id":
            item.get("fact_group_id"),

        "language":
            item.get("language"),

        "fact_type":
            item.get("fact_type"),

        "fact":
            item.get("fact"),

        "question":
            item.get("question"),

        "expected":
            expected,

        "predicted":
            predicted,

        "exact_match":
            correct
    })


# --------------------------------------------------
# Overall results
# --------------------------------------------------

total = len(results)

correct = sum(
    result["exact_match"]
    for result in results
)

accuracy = correct / total


print()
print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    "Total examples:",
    total
)

print(
    "Correct:",
    correct
)

print(
    "Accuracy:",
    f"{accuracy:.2%}"
)


# --------------------------------------------------
# Results by language
# --------------------------------------------------

print()
print("RESULTS BY LANGUAGE")
print("-" * 70)


languages = sorted(
    set(
        result["language"]
        for result in results
    )
)


for language in languages:

    language_results = [
        result
        for result in results
        if result["language"] == language
    ]

    language_correct = sum(
        result["exact_match"]
        for result in language_results
    )

    language_accuracy = (
        language_correct
        / len(language_results)
    )

    print(
        f"{language:12s} "
        f"{language_correct}/{len(language_results)} "
        f"({language_accuracy:.2%})"
    )


# --------------------------------------------------
# Results by fact type
# --------------------------------------------------

print()
print("RESULTS BY FACT TYPE")
print("-" * 70)


fact_types = sorted(
    set(
        result["fact_type"]
        for result in results
    )
)


for fact_type in fact_types:

    type_results = [
        result
        for result in results
        if result["fact_type"] == fact_type
    ]

    type_correct = sum(
        result["exact_match"]
        for result in type_results
    )

    type_accuracy = (
        type_correct
        / len(type_results)
    )

    print(
        f"{fact_type:15s} "
        f"{type_correct}/{len(type_results)} "
        f"({type_accuracy:.2%})"
    )


# --------------------------------------------------
# Save results
# --------------------------------------------------

with open(
    "data/baseline_results_chat.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print(
    "Saved results to "
    "data/baseline_results.json"
)
