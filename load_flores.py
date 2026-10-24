
from datasets import load_dataset

CONFIG = "hin_Deva-eng_Latn"

ds = load_dataset("facebook/flores", CONFIG)

dev = ds["dev"]
devtest = ds["devtest"]

print("Available splits:", list(ds.keys()))
print("Dev size:", len(dev))
print("Devtest size:", len(devtest))
print("Columns:", dev.column_names)
print("Features:", dev.features)

# Inspect an example before relying on column names
print("\nFirst example:")
print(dev[0])

# Confirm expected parallel sentence columns
expected = {"sentence_eng_Latn", "sentence_hin_Deva"}
missing = expected - set(dev.column_names)

if missing:
    raise ValueError(f"Missing columns: {missing}")

assert len(dev) > 0
assert len(devtest) > 0

# Check for missing or empty sentences
for split_name, split in [("dev", dev), ("devtest", devtest)]:
    for col in ["sentence_eng_Latn", "sentence_hin_Deva"]:
        empty = sum(
            not str(x).strip()
            for x in split[col]
        )
        print(f"{split_name} | {col} | empty={empty}")

# Parallel alignment sanity check
print("\nAlignment samples:")
for i in range(min(3, len(dev))):
    print(f"\nExample {i}")
    print("EN:", dev[i]["sentence_eng_Latn"])
    print("HI:", dev[i]["sentence_hin_Deva"])

print("\nFLORES loading and validation complete.")
