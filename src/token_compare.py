from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

texts = {
    "English": "The loan amount is ₹12,50,000.",
    "Hinglish": "Loan amount ₹12,50,000 hai.",
    "Hindi": "ऋण की राशि ₹12,50,000 है।",
}

for language, text in texts.items():

    encoded = tokenizer(
        text,
        return_tensors="pt"
    )

    token_ids = encoded["input_ids"][0]

    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print("\n" + "=" * 60)
    print(language)
    print("=" * 60)

    print("Text:")
    print(text)

    print("\nNumber of tokens:")
    print(len(token_ids))

    print("\nTokens:")
    for i, token in enumerate(tokens):
        print(i, repr(token))
