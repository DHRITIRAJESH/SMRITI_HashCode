import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BENCHMARK_PATH = "data/benchmark.json"

BUDGET_RATIO = 0.75
RECENT_TOKENS = 5
SINK_TOKENS = 4
MAX_NEW_TOKENS = 20


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

    # Repair the malformed beginning in memory only.
    repaired = "[\n{" + raw
    return json.loads(repaired)


def get_language(item):
    language = item.get("language")

    if language:
        return language

    fact = item.get("fact", "")

    if any("\u0900" <= c <= "\u097F" for c in fact):
        return "hi"

    return "unknown"


def check_value_recall(predicted, expected):
    return normalize(expected) in normalize(predicted)


def print_tokens(tokenizer, input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    print("\nPrompt tokens:")
    for i, token in enumerate(tokens):
        print(f"{i:3d}: {repr(token)}")


def aggregate_attention(attentions):
    """
    Convert attention tensors from:

        [batch, heads, query_length, key_length]

    into one importance score per cached token.

    We use the attention received by the newest query token.
    """

    total = None

    for attention in attentions:

        if attention is None:
            continue

        # [1, heads, query_length, key_length]
        attention = attention[0]

        # Attention from the newest query token.
        last_query = attention[:, -1, :]

        # Sum over attention heads.
        layer_score = last_query.sum(dim=0)

        if total is None:
            total = layer_score
        else:
            total = total + layer_score

    if total is None:
        raise RuntimeError("No attention tensors were returned.")

    return total.float()


class H2OCache:

    def __init__(
        self,
        max_cache_size,
        recent_tokens=5,
        sink_tokens=4,
    ):

        self.max_cache_size = max_cache_size
        self.recent_tokens = recent_tokens
        self.sink_tokens = sink_tokens

        self.cache = None

        # Original absolute position of every cached token.
        self.positions = None

        # Cumulative attention score for every cached token.
        self.scores = None

    def initialize(self, cache, scores):

        self.cache = cache

        seq_len = cache.get_seq_length()

        device = scores.device

        self.positions = torch.arange(
            seq_len,
            device=device,
            dtype=torch.long,
        )

        self.scores = scores.clone()

        print("\nH2O initialized")
        print("Initial cache size:", seq_len)
        print("Maximum cache size:", self.max_cache_size)

    def update_scores(self, new_scores):

        if new_scores.shape[0] != self.scores.shape[0]:
            raise RuntimeError(
                f"Attention/cache mismatch: "
                f"{new_scores.shape[0]} scores vs "
                f"{self.scores.shape[0]} cached tokens"
            )

        self.scores += new_scores

    def choose_keep_indices(self):

        n = self.positions.shape[0]

        if n <= self.max_cache_size:
            return torch.arange(
                n,
                device=self.positions.device,
            )

        recent_count = min(
            self.recent_tokens,
            self.max_cache_size,
            n,
        )

        sink_count = min(
            self.sink_tokens,
            self.max_cache_size - recent_count,
            n,
        )

        recent_start = n - recent_count

        sink_indices = torch.arange(
            sink_count,
            device=self.positions.device,
        )

        recent_indices = torch.arange(
            recent_start,
            n,
            device=self.positions.device,
        )

        protected = torch.cat(
            [sink_indices, recent_indices]
        ).unique()

        remaining_budget = self.max_cache_size - protected.numel()

        if remaining_budget <= 0:
            return torch.sort(protected).values

        all_indices = torch.arange(
            n,
            device=self.positions.device,
        )

        mask = torch.ones(
            n,
            dtype=torch.bool,
            device=self.positions.device,
        )

        mask[protected] = False

        candidates = all_indices[mask]

        candidate_scores = self.scores[candidates]

        _, top_local = torch.topk(
            candidate_scores,
            k=min(
                remaining_budget,
                candidates.numel(),
            ),
        )

        heavy_hitters = candidates[top_local]

        keep = torch.cat(
            [
                protected,
                heavy_hitters,
            ]
        )

        keep = torch.sort(keep).values

        return keep

    def evict(self):

        n = self.positions.shape[0]

        if n <= self.max_cache_size:
            return

        keep = self.choose_keep_indices()

        old_positions = self.positions.clone()

        self.cache = select_cache_positions(
            self.cache,
            keep,
        )

        self.positions = self.positions[keep]
        self.scores = self.scores[keep]

        print("\nH2O eviction")
        print("Before:", n)
        print("After :", len(keep))

        print(
            "Kept original positions:",
            self.positions.tolist(),
        )

        removed = sorted(
            set(old_positions.tolist())
            - set(self.positions.tolist())
        )

        print(
            "Removed original positions:",
            removed,
        )

    def append_token_cache(
        self,
        new_cache,
    ):

        """
        The model returns a cache containing the previous
        cache plus the newly generated token.

        We replace our cache with that cache and update
        bookkeeping.

        This method is intentionally separated because
        cache semantics need to be inspected carefully.
        """

        self.cache = new_cache

        new_length = new_cache.get_seq_length()

        expected_length = len(self.positions) + 1

        if new_length != expected_length:
            raise RuntimeError(
                f"Unexpected cache length. "
                f"Expected {expected_length}, "
                f"got {new_length}."
            )

        last_position = self.positions[-1].item() + 1

        self.positions = torch.cat(
            [
                self.positions,
                torch.tensor(
                    [last_position],
                    device=self.positions.device,
                    dtype=torch.long,
                ),
            ]
        )

        self.scores = torch.cat(
            [
                self.scores,
                torch.zeros(
                    1,
                    device=self.scores.device,
                    dtype=self.scores.dtype,
                ),
            ]
        )


def select_cache_positions(
    cache,
    indices,
):

    new_cache = DynamicCache()

    for layer_idx, layer in enumerate(cache.layers):

        keys = layer.keys
        values = layer.values

        layer_indices = indices.to(keys.device)

        new_keys = keys.index_select(
            2,
            layer_indices,
        )

        new_values = values.index_select(
            2,
            layer_indices,
        )

        new_cache.update(
            new_keys,
            new_values,
            layer_idx,
        )

    return new_cache


def generate_full_kv(
    model,
    tokenizer,
    inputs,
    max_new_tokens=20,
):

    print("\n==============================")
    print("FULL KV GENERATION")
    print("==============================")

    with torch.no_grad():

        outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )

    cache = outputs.past_key_values

    generated = []

    # First generated token comes directly from
    # the logits of the prefill.
    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    for _ in range(max_new_tokens):

        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)

        with torch.no_grad():

            outputs = model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

        cache = outputs.past_key_values

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return text.strip()


def generate_h2o(
    model,
    tokenizer,
    inputs,
    budget_ratio,
    max_new_tokens=20,
):

    print("\n==============================")
    print("H2O GENERATION")
    print("==============================")

    prompt_length = inputs["input_ids"].shape[1]

    budget = max(
        1,
        int(prompt_length * budget_ratio),
    )

    print("Prompt length:", prompt_length)
    print("Budget:", budget)

    # -------------------------------------------------
    # PREFILL
    # -------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    cache = outputs.past_key_values

    print(
        "Original cache length:",
        cache.get_seq_length(),
    )

    # -------------------------------------------------
    # Initial attention importance
    # -------------------------------------------------

    scores = aggregate_attention(
        outputs.attentions
    )

    print(
        "Initial score count:",
        len(scores),
    )

    # -------------------------------------------------
    # Create H2O cache manager
    # -------------------------------------------------

    h2o = H2OCache(
        max_cache_size=budget,
        recent_tokens=RECENT_TOKENS,
        sink_tokens=SINK_TOKENS,
    )

    h2o.initialize(
        cache,
        scores,
    )

    # -------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT feed the final prompt token again.
    #
    # The first generated token is obtained from the
    # logits produced by the prefill.
    # -------------------------------------------------

    next_token = torch.argmax(
        outputs.logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    generated = []

    for step in range(max_new_tokens):

        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated.append(token_id)

        print(
            f"\nGeneration step {step + 1}"
        )

        # -------------------------------------------------
        # Generate using current H2O cache
        # -------------------------------------------------

        current_position = (
            h2o.positions[-1].item() + 1
        )

        position_ids = torch.tensor(
            [[current_position]],
            device=next_token.device,
            dtype=torch.long,
        )

        cache_position = torch.tensor(
            [current_position],
            device=next_token.device,
            dtype=torch.long,
        )

        with torch.no_grad():

            outputs = model(
                input_ids=next_token,
                past_key_values=h2o.cache,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

        new_cache = outputs.past_key_values

        # -------------------------------------------------
        # Attention from newly generated token
        # -------------------------------------------------

        new_attention = aggregate_attention(
            outputs.attentions
        )

        # The returned attention includes the current
        # cached sequence. Its length should match the
        # cache BEFORE adding the new token.
        #
        # Depending on Transformers implementation,
        # attention may include the newly appended token.
        #
        # We therefore inspect the shape here.
        # -------------------------------------------------

        print(
            "Attention length:",
            len(new_attention),
        )

        print(
            "Cache returned by model:",
            new_cache.get_seq_length(),
        )

        # -------------------------------------------------
        # For this diagnostic implementation, the model
        # output cache already includes the newly generated
        # token.
        # -------------------------------------------------

        h2o.append_token_cache(
            new_cache
        )

        # -------------------------------------------------
        # Extend scores.
        #
        # The attention score corresponding to the new
        # token is not used as a historical score for
        # itself. We therefore add a zero score for it.
        #
        # Existing-token attention is added separately.
        # -------------------------------------------------

        old_count = len(h2o.scores) - 1

        usable_attention = new_attention[:old_count]

        h2o.scores[:old_count] += (
            usable_attention
        )

        # -------------------------------------------------
        # Evict if necessary
        # -------------------------------------------------

        h2o.evict()

        # -------------------------------------------------
        # Next token
        # -------------------------------------------------

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return text.strip()


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    model.eval()

    device = next(model.parameters()).device

    print("Device:", device)

    # -------------------------------------------------
    # Load exactly ONE benchmark example
    # -------------------------------------------------

    benchmark = load_benchmark(
        BENCHMARK_PATH
    )

    item = benchmark[0]

    prompt = item["fact"]
    expected = item["answer"]

    print("\n==============================")
    print("BENCHMARK CASE")
    print("==============================")

    print("Case:", item.get("case_id"))
    print("Language:", get_language(item))
    print("Fact:", prompt)
    print("Expected:", expected)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    print(
        "\nToken count:",
        inputs["input_ids"].shape[1],
    )

    print_tokens(
        tokenizer,
        inputs["input_ids"],
    )

    # -------------------------------------------------
    # FULL KV
    # -------------------------------------------------

    full_answer = generate_full_kv(
        model,
        tokenizer,
        inputs,
        MAX_NEW_TOKENS,
    )

    print("\nFULL KV ANSWER:")
    print(repr(full_answer))

    print(
        "Full KV recall:",
        check_value_recall(
            full_answer,
            expected,
        ),
    )

    # -------------------------------------------------
    # H2O
    # -------------------------------------------------

    h2o_answer = generate_h2o(
        model,
        tokenizer,
        inputs,
        BUDGET_RATIO,
        MAX_NEW_TOKENS,
    )

    print("\nH2O ANSWER:")
    print(repr(h2o_answer))

    print(
        "H2O recall:",
        check_value_recall(
            h2o_answer,
            expected,
        ),
    )


if __name__ == "__main__":
    main()
