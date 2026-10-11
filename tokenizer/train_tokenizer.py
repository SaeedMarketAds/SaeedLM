import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(BASE_DIR, "dataset", "raw", "saeed_corpus.txt")
TOKENIZER_OUTPUT = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")

print("🚀 Starting SaeedLM Tokenizer Training (HuggingFace BPE)...")

# 1. إعداد نموذج BPE
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 2. استخدام BertPreTokenizer لفصل علامات الترقيم
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

# 3. إعداد المدرب
trainer = trainers.BpeTrainer(
    vocab_size=500,
    min_frequency=1,
    special_tokens=[
        "[PAD]", "[UNK]", "[BOS]", "[EOS]",
        "[USER]", "[ASSISTANT]", "[SYSTEM]"
    ],
    show_progress=True
)

# 4. التدريب
print("📂 Corpus:", CORPUS)
tokenizer.train([CORPUS], trainer)

# 5. الحفظ
os.makedirs(os.path.dirname(TOKENIZER_OUTPUT), exist_ok=True)
tokenizer.save(TOKENIZER_OUTPUT)

print("✅ Tokenizer saved at:", TOKENIZER_OUTPUT)

# 6. اختبار فوري
print("\n🔎 Test:")
test_text = "[USER] ما هو التسويق الرقمي؟ [ASSISTANT]"
encoding = tokenizer.encode(test_text)
print("Text:  ", test_text)
print("Tokens:", encoding.tokens)
print("IDs:   ", encoding.ids)
print("\n📦 Vocabulary size:", tokenizer.get_vocab_size())
