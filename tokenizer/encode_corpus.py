import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORPUS = os.path.join(BASE_DIR, "dataset", "raw", "saeed_corpus.txt")
TOKENIZER = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
OUTPUT = os.path.join(BASE_DIR, "dataset", "raw", "token_ids.jsonl")

with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)

unk_id = vocab.get("[UNK]", 1)

print("🔤 SaeedLM Corpus Encoder (Pure Python v2)")
print("Corpus:", CORPUS)
print("Tokenizer:", TOKENIZER)
print("Output:", OUTPUT)
print()

TOKEN_PATTERN = re.compile(
    r'\[(?:USER|ASSISTANT|SYSTEM)\]'
    r'|[^\W\d_]+'
    r'|\d+'
    r'|[^\w\s]'
)

count = 0

with open(CORPUS, "r", encoding="utf-8") as src, \
     open(OUTPUT, "w", encoding="utf-8") as dst:

    for line in src:
        text = line.strip()
        if not text:
            continue

        tokens = TOKEN_PATTERN.findall(text)
        ids = [vocab.get(token, unk_id) for token in tokens]

        record = {
            "text": text,
            "tokens": tokens,
            "ids": ids
        }

        dst.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

        print("Text:", text)
        print("Tokens:", tokens)
        print("IDs:", ids)
        print()

print("✅ Encoding completed")
print("📦 Lines encoded:", count)
print("💾 Saved:", OUTPUT)
