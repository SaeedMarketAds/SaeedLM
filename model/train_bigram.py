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

transition = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float32)

with open(TOKEN_IDS, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        ids = record["ids"]

        sequence = [vocab["[BOS]"]] + ids + [vocab["[EOS]"]]

        for current, next_token in zip(sequence, sequence[1:]):
            transition[current, next_token] += 1

row_sums = transition.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1

transition /= row_sums

np.savez(MODEL_OUTPUT, transition=transition)

print("🧠 SaeedLM Bigram Model")
print("Vocabulary size:", VOCAB_SIZE)
print("Pairs:", int(transition.sum()))
print("✅ Model saved:", MODEL_OUTPUT)
