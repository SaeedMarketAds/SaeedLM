import json
import os
import re
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
MODEL = os.path.join(BASE_DIR, "model", "bigram_model.npz")

TOKEN_PATTERN = re.compile(
    r'\[(?:USER|ASSISTANT|SYSTEM)\]'
    r'|[^\W\d_]+'
    r'|\d+'
    r'|[^\w\s]'
)

with open(TOKENIZER, "r", encoding="utf-8") as f:
    vocab = json.load(f)

id_to_token = {v: k for k, v in vocab.items()}
transition = np.load(MODEL)["transition"]

VOCAB_SIZE = len(vocab)
if transition.shape != (VOCAB_SIZE, VOCAB_SIZE):
    raise ValueError(
        f"Model shape {transition.shape} does not match vocabulary {VOCAB_SIZE}"
    )

UNK = vocab["[UNK]"]
BOS = vocab["[BOS]"]
EOS = vocab["[EOS]"]
ASSISTANT = vocab["[ASSISTANT]"]

print("SaeedLM Bigram Generator (v2)")
print("Vocabulary size:", VOCAB_SIZE)
print("Model shape:", transition.shape)
print("Type exit to quit.")
print()

rng = np.random.default_rng()

while True:
    try:
        question = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nSaeedLM: إلى اللقاء 👋")
        break
    if not question:
        continue
    if question.lower() == "exit":
        print("Bye.")
        break

    normalized = question.strip().lower()
    normalized = normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    normalized = normalized.replace("ى", "ي")
    normalized = re.sub(r"[؟?!.,،؛:]+", "", normalized)
    normalized = " ".join(normalized.split())

    greetings = {
        "السلام عليكم": "وعليكم السلام ورحمة الله وبركاته، أهلاً بك!",
        "سلام عليكم": "وعليكم السلام ورحمة الله وبركاته، أهلاً بك!",
        "مرحبا": "مرحباً بك! كيف أستطيع مساعدتك؟",
        "اهلا": "أهلاً وسهلاً بك!",
        "صباح الخير": "صباح النور والسرور!",
        "مساء الخير": "مساء النور والسرور!",
        "كيف حالك": "أنا SaeedLM، جاهز لمساعدتك!",
        "من انت": "أنا SaeedLM، نموذج لغوي تجريبي قيد التطوير.",
        "من الذي برمجك": "طوّر سعيد المسوري مشروع SaeedLM، وهو نموذج تجريبي قيد التطوير.",
        "من برمجك": "طوّر سعيد المسوري مشروع SaeedLM.",
        "تعرف سعيد المسوري": "سعيد المسوري هو صاحب مشروع SaeedLM ومطوّر المشروع.",
    }

    if normalized in greetings:
        print("SaeedLM:", greetings[normalized])
        print()
        continue

    tokens = TOKEN_PATTERN.findall(question)
    ids = [vocab.get(token, UNK) for token in tokens]
    unknown = sum(token_id == UNK for token_id in ids)
    print("Input tokens:", tokens)
    print("Unknown input tokens:", unknown)

    current = ASSISTANT
    generated = []

    for _ in range(40):
        probs = transition[current]
        total = float(probs.sum())
        if total <= 0:
            break

        next_token = int(rng.choice(VOCAB_SIZE, p=probs / total))
        token = id_to_token[next_token]

        if next_token == EOS or token in ("[BOS]", "[USER]", "[SYSTEM]"):
            break
        if next_token == UNK:
            break
        if next_token == ASSISTANT:
            break

        generated.append(token)
        current = next_token

    print("SaeedLM:", " ".join(generated))
    print()
