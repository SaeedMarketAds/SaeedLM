
import json
import re
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"
MODEL_PATH = ROOT / "model" / "bigram_model.npz"

TOKEN_PATTERN = re.compile(
    r'\[(?:USER|ASSISTANT|SYSTEM)\]'
    r'|[^\W\d_]+'
    r'|\d+'
    r'|[^\w\s]',
    re.UNICODE,
)

GREETINGS = {
    "السلام عليكم": "وعليكم السلام ورحمة الله وبركاته، أهلاً بك!",
    "سلام عليكم": "وعليكم السلام ورحمة الله وبركاته، أهلاً بك!",
    "مرحبا": "مرحباً بك! كيف أستطيع مساعدتك؟",
    "اهلا": "أهلاً وسهلاً بك!",
    "صباح الخير": "صباح النور والسرور!",
    "مساء الخير": "مساء النور والسرور!",
    "كيف حالك": "أنا SaeedLM، جاهز لمساعدتك!",
    "من انت": "أنا SaeedLM، نموذج لغوي تجريبي قيد التطوير.",
    "من الذي برمجك": "طوّر سعيد المسوري مشروع SaeedLM.",
    "من برمجك": "طوّر سعيد المسوري مشروع SaeedLM.",
    "تعرف سعيد المسوري": "سعيد المسوري هو صاحب مشروع SaeedLM ومطوّره.",
}


def normalize(text):
    text = text.strip().lower()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = re.sub(r"[؟?!.,،؛:]+", "", text)
    return " ".join(text.split())


st.set_page_config(
    page_title="SaeedLM",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
    <div style="text-align:center;padding:18px;
                background:#07111f;border:1px solid #ffc52f;
                border-radius:14px;">
      <h1 style="color:#ffc52f;">🧠 SaeedLM</h1>
      <p style="color:#00e5ff;">نموذج لغوي عربي تجريبي مستقل</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "نسخة تجريبية مبنية على نموذج Bigram؛ "
    "قد تكون بعض الإجابات قصيرة أو غير مترابطة."
)


@st.cache_resource
def load_model():
    with TOKENIZER_PATH.open(encoding="utf-8") as f:
        vocab = json.load(f)

    with np.load(MODEL_PATH) as data:
        transition = data["transition"].astype(np.float64)

    required = ["[UNK]", "[BOS]", "[EOS]", "[USER]", "[ASSISTANT]"]
    if not all(token in vocab for token in required):
        raise ValueError("بعض الرموز الأساسية غير موجودة في tokenizer.")

    if transition.shape != (len(vocab), len(vocab)):
        raise ValueError("أبعاد النموذج لا تطابق tokenizer.")

    if not np.isfinite(transition).all():
        raise ValueError("توجد قيم غير صالحة في مصفوفة النموذج.")

    id_to_token = {int(idx): token for token, idx in vocab.items()}
    return vocab, id_to_token, transition


def generate_reply(question, vocab, id_to_token, transition):
    normalized = normalize(question)

    if normalized in GREETINGS:
        return GREETINGS[normalized]

    tokens = TOKEN_PATTERN.findall(question)
    if not tokens:
        return "اكتب سؤالًا أو عبارة أولًا."

    ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
    current = ids[-1]
    generated = []
    rng = np.random.default_rng()

    blocked = [
        vocab[token]
        for token in ("[BOS]", "[USER]", "[ASSISTANT]", "[UNK]", "[SYSTEM]")
        if token in vocab
    ]
    eos = vocab["[EOS]"]

    for _ in range(35):
        probs = transition[current].copy()
        probs[blocked] = 0.0

        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            break

        current = int(rng.choice(len(probs), p=probs / total))

        if current == eos:
            break

        token = id_to_token.get(current, "")
        if token and not token.startswith("["):
            generated.append(token)

    answer = " ".join(generated).strip()
    return answer or "لم يتمكن النموذج من توليد إجابة. جرّب عبارة أخرى."


try:
    vocab, id_to_token, transition = load_model()
except Exception as exc:
    st.error(f"تعذر تحميل النموذج: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("مسح المحادثة"):
    st.session_state.messages = []
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("اكتب رسالتك إلى SaeedLM...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    answer = generate_reply(
        question, vocab, id_to_token, transition
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)

st.caption(
    "SaeedLM نموذج Bigram تجريبي صغير. "
    "هذه الواجهة لا تعيد تدريب النموذج."
)
