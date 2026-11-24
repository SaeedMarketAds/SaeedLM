import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

TOKEN_PATTERN = re.compile(
    r'\[(?:USER|ASSISTANT|SYSTEM)\]|[^\W\d_]+|\d+|[^\w\s]',
    re.UNICODE
)

def load_model(tokenizer_path, model_path):
    with open(tokenizer_path, encoding="utf-8") as f:
        vocab = json.load(f)

    with np.load(model_path, allow_pickle=False) as d:
        transition = d["transition"].astype(np.float64)

    if transition.shape != (len(vocab), len(vocab)):
        raise ValueError(
            f"Vocabulary/model mismatch: {len(vocab)} vs {transition.shape}"
        )

    return vocab, {int(v): k for k, v in vocab.items()}, transition

def generate(question, vocab, id_to_token, transition):
    rng = np.random.default_rng(42)
    tokens = TOKEN_PATTERN.findall(question)

    if not tokens:
        return "", 0, 0

    unk = vocab.get("[UNK]")
    if unk is None:
        raise ValueError("Tokenizer missing [UNK]")

    ids = [vocab.get(t, unk) for t in tokens]
    blocked = [
        vocab[t] for t in ("[BOS]", "[USER]", "[SYSTEM]", "[UNK]")
        if t in vocab
    ]
    eos = vocab.get("[EOS]")
    current = ids[-1]
    generated = []

    for _ in range(35):
        probs = transition[current].copy()

        for b in blocked:
            if b < len(probs):
                probs[b] = 0

        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            break

        probs /= total
        nxt = int(rng.choice(len(probs), p=probs))

        if nxt == eos:
            break

        tok = id_to_token.get(nxt, "")
        if tok and not tok.startswith("["):
            generated.append(tok)

        current = nxt

    return " ".join(generated), len(ids), ids.count(unk)

def metrics(text):
    words = text.split()
    return {
        "length": len(words),
        "unique": len(set(words)),
        "repeat_pct": round(
            (len(words) - len(set(words))) / len(words) * 100, 1
        ) if words else 0
    }

def main():
    models = {
        "OLD": load_model(
            ROOT / "tokenizer/tokenizer.json",
            ROOT / "model/bigram_model.npz"
        ),
        "A8": load_model(
            ROOT / "tokenizer/tokenizer_a8_stable.json",
            ROOT / "model/bigram_model_a8.npz"
        )
    }

    questions = [
        "ما هو التسويق الرقمي؟",
        "كيف أنجح في الإعلانات الممولة؟",
        "ما أهمية تحسين محركات البحث SEO؟",
        "كيف أزيد مبيعات المنتجات؟"
    ]

    for q in questions:
        print("\n" + "=" * 65)
        print("QUESTION:", q)

        for name, (vocab, id_to_token, transition) in models.items():
            answer, count, unk = generate(
                q, vocab, id_to_token, transition
            )
            print(f"\n[{name}]")
            print("Input tokens:", count, "| UNK:", unk)
            print("Answer:", answer or "(empty)")
            print("Metrics:", metrics(answer))

    print("\nEvaluation complete. No model or data files modified.")

if __name__ == "__main__":
    main()
