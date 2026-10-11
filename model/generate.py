import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
MODEL = os.path.join(BASE_DIR, "model", "bigram_model.npz")

with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)

id_to_token = {v: k for k, v in vocab.items()}

transition = np.load(MODEL)["transition"]

print("🤖 SaeedLM Bigram Generator")
print("اكتب سؤالك، أو اكتب exit للخروج.")

while True:
    question = input("\nأنت: ").strip()

    if question.lower() == "exit":
        print("👋 تم الخروج.")
        break

    words = question.split()
    ids = [vocab.get(word, vocab["[UNK]"]) for word in words]

    current = vocab["[ASSISTANT]"]
    generated = []
    started_answer = True

    for _ in range(30):
        probs = transition[current]

        if probs.sum() == 0:
            break

        next_token = np.random.choice(len(probs), p=probs)
        token = id_to_token[next_token]

        if token == "[ASSISTANT]":
            started_answer = True
            current = next_token
            continue

        if token in ("[EOS]", "[USER]", "[SYSTEM]", "[UNK]"):
            break

        if started_answer:
            generated.append(token)

        current = next_token

    print("SaeedLM:", " ".join(generated))
