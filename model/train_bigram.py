import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
TOKEN_IDS = os.path.join(BASE_DIR, "dataset", "raw", "token_ids.jsonl")
MODEL_OUTPUT = os.path.join(BASE_DIR, "model", "bigram_model.npz")

with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)
assert "[BOS]" in vocab and "[EOS]" in vocab, "Missing BOS/EOS tokens"

transition = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)
pair_count = 0

with open(TOKEN_IDS, "r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        ids = record["ids"]

        for token_id in ids:
            if not isinstance(token_id, int) or not 0 <= token_id < VOCAB_SIZE:
                raise ValueError(
                    f"Invalid token ID at line {line_number}: {token_id}"
                )

        sequence = [vocab["[BOS]"]] + ids + [vocab["[EOS]"]]

        for current, next_token in zip(sequence, sequence[1:]):
            transition[current, next_token] += 1
            pair_count += 1

row_sums = transition.sum(axis=1, keepdims=True)
nonzero_rows = int((row_sums[:, 0] > 0).sum())
row_sums[row_sums == 0] = 1
transition /= row_sums

if transition.shape != (VOCAB_SIZE, VOCAB_SIZE):
    raise ValueError("Unexpected transition matrix shape")
if not np.isfinite(transition).all():
    raise ValueError("Model contains non-finite values")

np.savez(MODEL_OUTPUT, transition=transition)

print("SaeedLM Bigram Model")
print("Vocabulary size:", VOCAB_SIZE)
print("Model shape:", transition.shape)
print("Observed pairs:", pair_count)
print("Non-zero rows:", nonzero_rows)
print("Model saved:", MODEL_OUTPUT)
