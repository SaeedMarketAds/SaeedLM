import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer_a8_stable.json"
TOKEN_IDS_PATH = ROOT / "dataset" / "raw" / "token_ids_a8.jsonl"
MODEL_OUTPUT = ROOT / "model" / "bigram_model_a8.npz"


def load_vocab():
    with TOKENIZER_PATH.open(encoding="utf-8") as file:
        vocab = json.load(file)

    if not isinstance(vocab, dict):
        raise ValueError("Invalid tokenizer")

    if len(vocab) != len(set(vocab.values())):
        raise ValueError("Duplicate token IDs")

    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Token IDs are not continuous")

    for token in ("[BOS]", "[EOS]", "[UNK]"):
        if token not in vocab:
            raise ValueError(f"Missing token: {token}")

    return vocab


def train(vocab):
    size = len(vocab)
    transition = np.zeros((size, size), dtype=np.float64)

    pair_count = 0
    line_count = 0

    with TOKEN_IDS_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            ids = record.get("ids")

            if not isinstance(ids, list):
                raise ValueError(f"Invalid IDs at line {line_number}")

            for token_id in ids:
                if (
                    not isinstance(token_id, int)
                    or isinstance(token_id, bool)
                    or not 0 <= token_id < size
                ):
                    raise ValueError(f"Invalid token ID: {token_id}")

            sequence = [vocab["[BOS]"]] + ids + [vocab["[EOS]"]]

            for current, next_token in zip(sequence, sequence[1:]):
                transition[current, next_token] += 1
                pair_count += 1

            line_count += 1

    if pair_count == 0:
        raise ValueError("No training pairs found")

    row_sums = transition.sum(axis=1, keepdims=True)
    nonzero_rows = int((row_sums[:, 0] > 0).sum())

    row_sums[row_sums == 0] = 1
    transition /= row_sums

    if not np.isfinite(transition).all():
        raise ValueError("Non-finite model values")

    if MODEL_OUTPUT.exists():
        raise FileExistsError("A8 model already exists. Refusing overwrite.")

    np.savez_compressed(
        MODEL_OUTPUT,
        transition=transition.astype(np.float32),
        vocab_size=np.array(size),
        pair_count=np.array(pair_count),
        line_count=np.array(line_count),
    )

    print("A8 training complete")
    print("Vocabulary:", size)
    print("Matrix:", transition.shape)
    print("Training lines:", line_count)
    print("Observed pairs:", pair_count)
    print("Non-zero rows:", nonzero_rows)
    print("Model:", MODEL_OUTPUT)


if __name__ == "__main__":
    vocabulary = load_vocab()
    train(vocabulary)
