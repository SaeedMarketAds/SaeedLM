import json
import os
import re
from collections import Counter

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CORPUS = os.path.join(
    BASE_DIR, "dataset", "raw", "saeed_corpus.txt"
)

OUTPUT = os.path.join(
    BASE_DIR, "tokenizer", "tokenizer.json"
)

TOKEN_PATTERN = re.compile(
    r'\[(?:USER|ASSISTANT|SYSTEM)\]'
    r'|[^\W\d_]+'
    r'|\d+'
    r'|[^\w\s]',
    re.UNICODE
)

SPECIAL = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]",
    "[USER]", "[ASSISTANT]", "[SYSTEM]"
]

COMMON = [
    "من", "أنت", "أنا", "السلام", "عليكم",
    "حالك", "حالكم", "هل", "ماذا", "لماذا",
    "أين", "متى", "كيف", "كم", "نعم", "لا",
    "شكرا", "أهلا", "مرحبا", "وداعا",
    "ما", "هو", "هي", "في", "على", "إلى",
    "عن", "و", "التسويق", "الرقمي",
    "الإعلانات", "الممولة", "SEO", "المحتوى",
    "العلامة", "التجارية", "الجمهور",
    "المستهدف", "؟", "،", ".", "!", ":", "؛"
]


def main():
    if not os.path.isfile(CORPUS):
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS}"
        )

    counter = Counter()

    with open(CORPUS, encoding="utf-8") as file:
        for line in file:
            counter.update(TOKEN_PATTERN.findall(line.strip()))

    vocab = {}

    for token in SPECIAL + COMMON:
        if token not in vocab:
            vocab[token] = len(vocab)

    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)

    print("SaeedLM Pure Python Tokenizer")
    print("Unique corpus tokens:", len(counter))
    print("Total corpus tokens:", sum(counter.values()))
    print("Vocabulary size:", len(vocab))
    print("Question mark ID:", vocab.get("؟", "MISSING"))
    print("SEO ID:", vocab.get("SEO", "MISSING"))

    # Write to a separate file first; do not replace the active vocabulary.
    candidate = os.path.join(
        BASE_DIR, "tokenizer", "tokenizer_candidate.json"
    )

    with open(candidate, "w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)

    # Validate the saved candidate.
    with open(candidate, encoding="utf-8") as file:
        check = json.load(file)

    assert check == vocab
    assert all(token in check for token in SPECIAL)
    assert "؟" in check
    assert "SEO" in check

    print("Candidate saved:", candidate)
    print("Validation: OK")
    print("Active tokenizer.json was not changed.")


if __name__ == "__main__":
    main()
