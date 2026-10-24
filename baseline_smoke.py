
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to("cuda:0")

model.eval()
print("Device:", model.device)


def run_case(language="en"):
    if language == "en":
        fact = "The reference amount is ₹7,43,281."
        question = "What is the reference amount?"
    else:
        fact = "संदर्भ राशि ₹7,43,281 है।"
        question = "संदर्भ राशि क्या है?"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an exact information retrieval system. "
                "Answer using only the record. "
                "Return only the requested value, "
                "without explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Record:\n{fact}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    print("\n" + "=" * 50)
    print("LANGUAGE:", language)
    print("INPUT TOKENS:", inputs["input_ids"].shape[1])

    print("PROMPT:")
    print(tokenizer.decode(
        inputs["input_ids"][0],
        skip_special_tokens=False,
    ))

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens.
    generated = output[0, inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    print("EXPECTED:", "₹7,43,281")
    print("ANSWER:", repr(answer))

    # Conservative exact-answer test.
    # Don't accept an answer merely because it contains
    # the correct amount somewhere in a long explanation.
    normalized_answer = answer.replace(" ", "").strip()
    normalized_expected = "₹7,43,281"

    correct = normalized_answer == normalized_expected
    print("EXACT MATCH:", correct)

    return correct


if __name__ == "__main__":
    run_case("en")
    run_case("hi")
