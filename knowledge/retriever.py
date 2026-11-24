"""SaeedLM knowledge retrieval with confidence threshold."""

import json
import re
import sys
from pathlib import Path

KB_PATH = Path(__file__).resolve().parent / "knowledge.json"


def normalize(text):
    text = str(text or "").lower()
    text = text.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ى": "ي", "ئ": "ي", "ؤ": "و",
        "ة": "ه",
    }))
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return " ".join(text.split())


class KnowledgeBase:
    def __init__(self, path=KB_PATH):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.entries = data["knowledge_base"]["entries"]

    def _score(self, entry, query, tokens):
        title = normalize(entry.get("title", ""))
        content = normalize(entry.get("content", ""))
        keywords = [normalize(k) for k in entry.get("keywords", [])]
        score = 0.0
        matched = []

        for keyword in keywords:
            if not keyword:
                continue
            if keyword in query:
                score += 10
                matched.append(keyword)
            elif set(keyword.split()).issubset(tokens):
                score += 5
                matched.append(keyword)

        score += sum(
            2 for token in tokens
            if len(token) > 1 and token in title
        )
        score += sum(
            1 for token in tokens
            if len(token) > 2 and token in content
        )

        if score:
            score += float(entry.get("priority", 0)) / 1000

        return score, matched

    def search(self, query, top_k=3, min_score=5.0):
        normalized_query = normalize(query)
        if not normalized_query:
            return []

        tokens = set(normalized_query.split())
        results = []

        for entry in self.entries:
            score, matched = self._score(
                entry, normalized_query, tokens
            )
            if score >= min_score:
                results.append((score, entry, matched))

        results.sort(key=lambda item: item[0], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    kb = KnowledgeBase()
    print("Knowledge entries:", len(kb.entries))

    query = " ".join(sys.argv[1:])
    if not query:
        sys.exit(0)

    results = kb.search(query, top_k=3, min_score=5.0)

    if not results:
        print("No sufficiently strong knowledge match.")
        sys.exit(0)

    for index, (score, entry, matched) in enumerate(results, 1):
        print(f"\n[{index}] {entry.get('title', '')}")
        print("ID:", entry.get("id", ""))
        print(f"Score: {score:.2f}")
        print("Matched keywords:", matched)
        print("Content:", entry.get("content", "")[:250])
