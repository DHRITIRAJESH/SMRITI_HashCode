import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()
print("Model loaded successfully.")

# Check where the model was placed
print("Model device:", model.get_input_embeddings().weight.device)
print("CUDA available:", torch.cuda.is_available())

prompt = "The loan amount is ₹12,50,000."

inputs = tokenizer(prompt, return_tensors="pt").to(
    model.get_input_embeddings().weight.device
)

with torch.inference_mode():
    outputs = model(**inputs, use_cache=True)

print("\nForward pass successful.")
print("Input tokens:", inputs["input_ids"].shape[1])
print("Output logits shape:", outputs.logits.shape)
print("KV cache available:", outputs.past_key_values is not None)

cache = outputs.past_key_values

print("Cache type:", type(cache))

# Inspect the modern Transformers Cache API
if hasattr(cache, "layers"):
    keys = [layer.keys for layer in cache.layers]
    values = [layer.values for layer in cache.layers]
elif hasattr(cache, "key_cache"):
    # Compatibility with older Transformers versions
    keys = cache.key_cache
    values = cache.value_cache
else:
    raise RuntimeError(
        f"Unsupported KV cache type: {type(cache)}"
    )

print("Number of layers:", len(keys))

for layer_idx in range(min(3, len(keys))):
    print(
        f"Layer {layer_idx}: "
        f"K shape = {tuple(keys[layer_idx].shape)}, "
        f"V shape = {tuple(values[layer_idx].shape)}"
    )

# Generate a short continuation to verify decoding
with torch.inference_mode():
    generated = model.generate(
        **inputs,
        max_new_tokens=20,
        use_cache=True,
        do_sample=False
    )

print("\nGenerated text:")
print(tokenizer.decode(generated[0], skip_special_tokens=True))

print("\nKV cache inspection and generation successful.")


