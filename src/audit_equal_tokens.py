
import torch
import csv
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

HINDI_PATH = RESULTS / "bases_hindi.pt"
ENGLISH_PATH = RESULTS / "bases_english.pt"

OUTPUT_PATH = RESULTS / "equal_token_rank_audit.csv"

N = 20000
SEEDS = list(range(10))

LANGUAGES = ["hindi", "english"]
THRESHOLDS = [0.90, 0.95, 0.99]

# --------------------------------------------------
# Load fitted bases
# --------------------------------------------------

def load_bases(path):
    print(f"Loading {path}")
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


bases = {
    "hindi": load_bases(HINDI_PATH),
    "english": load_bases(ENGLISH_PATH),
}

for lang in LANGUAGES:
    meta = bases[lang]["metadata"]
    print(f"\n{lang.upper()} metadata:")
    print(meta)

# --------------------------------------------------
# Load activations
# --------------------------------------------------

def load_activations(path):
    print(f"\nLoading {path}")
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


activations = {
    "hindi": load_activations(
        RESULTS / "activations_hindi.pt"
    ),
    "english": load_activations(
        RESULTS / "activations_english.pt"
    ),
}

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_collection(data, kind):
    """
    Expected activation structure:
    data[kind][layer][head]

    kind: keys or values
    """

    collection = data[kind]

    return collection


def get_token_count(collection):
    """
    Assumes each head activation has shape:
    [features, tokens]

    Checks all heads/layers have consistent token count.
    """

    counts = set()

    for layer in collection:
        for head in layer:
            counts.add(head.shape[1])

    if len(counts) != 1:
        raise ValueError(
            f"Inconsistent token counts: {counts}"
        )

    return counts.pop()


# --------------------------------------------------
# Verify sample size
# --------------------------------------------------

collections = {}

for lang in LANGUAGES:
    collections[lang] = {
        "keys": get_collection(
            activations[lang], "keys"
        ),
        "values": get_collection(
            activations[lang], "values"
        ),
    }

token_counts = {}

for lang in LANGUAGES:
    token_counts[lang] = get_token_count(
        collections[lang]["keys"]
    )

    value_count = get_token_count(
        collections[lang]["values"]
    )

    if value_count != token_counts[lang]:
        raise ValueError(
            f"{lang}: Key/Value token counts differ"
        )

    print(
        f"{lang}: {token_counts[lang]:,} tokens"
    )

assert all(
    token_counts[lang] >= N
    for lang in LANGUAGES
), "Not enough tokens for requested sample size"

print(f"\nEqual-token sample size: {N:,}")
print(f"Seeds: {SEEDS}")

# --------------------------------------------------
# Compute ranks from sampled activations
# --------------------------------------------------

def calculate_rank(X, threshold):
    """
    X: [features, sampled_tokens]

    Center the sampled vectors and calculate
    covariance eigenvalues.

    Returns minimum number of principal components
    needed to explain the requested variance.
    """

    X = X.float()

    # [features, tokens]
    X = X - X.mean(dim=1, keepdim=True)

    # Covariance matrix
    covariance = X @ X.T

    covariance /= max(X.shape[1] - 1, 1)

    # Symmetric covariance matrix
    eigenvalues = torch.linalg.eigvalsh(
        covariance
    )

    eigenvalues = torch.clamp(
        eigenvalues, min=0
    )

    eigenvalues = torch.sort(
        eigenvalues, descending=True
    ).values

    total = eigenvalues.sum()

    if total <= 0:
        return 0

    cumulative = torch.cumsum(
        eigenvalues, dim=0
    ) / total

    rank = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(
                threshold,
                dtype=cumulative.dtype
            )
        ).item()
    ) + 1

    return min(rank, len(eigenvalues))


# --------------------------------------------------
# Run audit
# --------------------------------------------------

rows = []

for seed in SEEDS:
    print(f"\nRunning seed {seed}")

    sampled = {}

    # Independently sample BOTH languages.
    for lang in LANGUAGES:
        sampled[lang] = {}

        generator = torch.Generator()
        generator.manual_seed(seed + (
            100000 if lang == "english" else 0
        ))

        total_tokens = token_counts[lang]

        indices = torch.randperm(
            total_tokens,
            generator=generator
        )[:N]

        indices = indices.sort().values

        for kind in ["keys", "values"]:
            sampled[lang][kind] = []

            for layer in collections[lang][kind]:
                sampled_layer = []

                for head in layer:
                    X = head[:, indices]

                    sampled_layer.append(X)

                sampled[lang][kind].append(
                    sampled_layer
                )

    # Compare ranks
    for kind in ["keys", "values"]:
        for layer_idx in range(
            len(sampled["hindi"][kind])
        ):
            hindi_layer = sampled["hindi"][kind][
                layer_idx
            ]

            english_layer = sampled["english"][kind][
                layer_idx
            ]

            for head_idx in range(
                len(hindi_layer)
            ):
                X_h = hindi_layer[head_idx]
                X_e = english_layer[head_idx]

                for threshold in THRESHOLDS:
                    rank_h = calculate_rank(
                        X_h, threshold
                    )

                    rank_e = calculate_rank(
                        X_e, threshold
                    )

                    rows.append({
                        "seed": seed,
                        "kind": kind,
                        "layer": layer_idx,
                        "head": head_idx,
                        "threshold": threshold,
                        "n_tokens": N,
                        "hindi_rank": rank_h,
                        "english_rank": rank_e,
                        "difference_hindi_minus_english":
                            rank_h - rank_e,
                    })

    print(f"Finished seed {seed}")

# --------------------------------------------------
# Save results
# --------------------------------------------------

fieldnames = list(rows[0].keys())

with open(
    OUTPUT_PATH,
    "w",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved: {OUTPUT_PATH}")
print(f"Total rows: {len(rows)}")

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n========== SUMMARY ==========")

summary = defaultdict(list)

for row in rows:
    key = (
        row["kind"],
        row["threshold"]
    )

    summary[key].append(
        row["difference_hindi_minus_english"]
    )

for (kind, threshold), diffs in sorted(
    summary.items()
):
    mean_diff = sum(diffs) / len(diffs)

    higher = sum(d > 0 for d in diffs)
    lower = sum(d < 0 for d in diffs)
    equal = sum(d == 0 for d in diffs)

    print(f"\n{kind.upper()} | {threshold:.0%}")
    print(f"Mean difference: {mean_diff:.4f}")
    print(f"Hindi higher: {higher}")
    print(f"Hindi lower: {lower}")
    print(f"Equal: {equal}")
    print(f"Comparisons: {len(diffs)}")

    # Per-seed average differences
    print("Per-seed mean differences:")

    for seed in SEEDS:
        seed_diffs = [
            row["difference_hindi_minus_english"]
            for row in rows
            if row["seed"] == seed
            and row["kind"] == kind
            and row["threshold"] == threshold
        ]

        seed_mean = (
            sum(seed_diffs) / len(seed_diffs)
        )

        print(
            f"  Seed {seed}: {seed_mean:.4f}"
        )

print("\nAudit complete.")
