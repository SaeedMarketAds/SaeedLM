
import base64
import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer_a8_stable.json"
MODEL_PATH = ROOT / "model" / "bigram_model_a8.npz"

AVATAR_CANDIDATES = [
    ROOT / "saeedlm_avatar.png",
    ROOT / "saeedlm_robot.png",
]

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
    text = str(text).strip().lower()
    text = re.sub(r"[\\u064B-\\u065F\\u0670\\u0640]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ٱ", "ا").replace("ى", "ي")
    text = re.sub(r"[؟?!.,،؛:…]+", " ", text)
    return " ".join(text.split())

st.set_page_config(
    page_title="SaeedLM",
    page_icon="🧠",
    layout="centered",
)

avatar_path = next(
    (p for p in AVATAR_CANDIDATES if p.is_file()),
    None
)

voice_slot = st.empty()

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

    for greeting, reply in sorted(
        GREETINGS.items(),
        key=lambda item: len(normalize(item[0])),
        reverse=True,
    ):
        phrase = normalize(greeting)
        if normalized == phrase or normalized.startswith(phrase + " "):
            return reply

    tokens = TOKEN_PATTERN.findall(question)
    if not tokens:
        return "اكتب سؤالًا أو عبارة أولًا."

    ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
    current = ids[-1]
    generated = []
    rng = np.random.default_rng()

    blocked = [
        vocab[token]
        for token in ("[BOS]", "[USER]", "[UNK]", "[SYSTEM]")
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


# ===== SAeedLM VOICE + INTERACTIVE LIGHT =====
latest_answer = next(
    (
        m["content"]
        for m in reversed(st.session_state.messages)
        if m["role"] == "assistant"
    ),
    ""
)

if avatar_path and avatar_path.is_file():
    avatar_b64 = base64.b64encode(
        avatar_path.read_bytes()
    ).decode("ascii")

    voice_html = """
    <html>
    <head>
    <style>
      body {
        margin: 0;
        background: transparent;
        font-family: sans-serif;
        color: white;
      }

      .panel {
        text-align: center;
        padding: 16px;
        border: 1px solid #b89535;
        border-radius: 18px;
        background: #07111f;
      }

      .avatar {
        width: 145px;
        height: 145px;
        object-fit: cover;
        border-radius: 50%;
        border: 3px solid #168bff;
        box-shadow: 0 0 15px #087bff;
        transition: .3s;
      }

      .speaking .avatar {
        animation: glow 1.2s infinite alternate;
      }

      @keyframes glow {
        from {
          box-shadow: 0 0 12px #00cfff,
                      0 0 25px #087bff;
          border-color: #00e5ff;
          transform: scale(1);
        }
        to {
          box-shadow: 0 0 25px #00e5ff,
                      0 0 55px #d4af37;
          border-color: #ffd700;
          transform: scale(1.035);
        }
      }

      h3 {
        color: #ffc52f;
        margin: 12px;
      }

      button {
        border: 1px solid #00cfff;
        border-radius: 12px;
        padding: 11px 17px;
        margin: 5px;
        color: white;
        background: #102c49;
        font-size: 15px;
      }

      button:disabled {
        opacity: .4;
      }
    </style>
    </head>

    <body>
      <div class="panel" id="panel">
        <img class="avatar" src="data:image/png;base64,__IMAGE__">
        <h3>🤖 SaeedLM</h3>
        <p id="status">جاهز للتحدث معك</p>

        <button id="speak" __DISABLED__>🔊 استمع إلى الإجابة</button>
        <button id="stop">⏹ إيقاف الصوت</button>
      </div>

      <script>
        const answer = __TEXT__;
        const panel = document.getElementById("panel");
        const status = document.getElementById("status");

        const speak = document.getElementById("speak");
        const stop = document.getElementById("stop");

        speak.onclick = function() {
          if (!answer || !window.speechSynthesis) {
            status.textContent = "الصوت غير متاح في المتصفح";
            return;
          }

          speechSynthesis.cancel();

          const utterance = new SpeechSynthesisUtterance(answer);
          utterance.lang = "ar-SA";
          utterance.rate = 0.92;
          utterance.pitch = 1.03;

          const voices = speechSynthesis.getVoices();
          const arabicVoice = voices.find(
            v => v.lang.toLowerCase().startsWith("ar")
          );

          if (arabicVoice) {
            utterance.voice = arabicVoice;
          }

          utterance.onstart = function() {
            panel.classList.add("speaking");
            status.textContent = "🟢 سعيد يتحدث الآن...";
          };

          utterance.onend = function() {
            panel.classList.remove("speaking");
            status.textContent = "اكتمل الحديث";
          };

          utterance.onerror = function() {
            panel.classList.remove("speaking");
            status.textContent = "تعذر تشغيل الصوت";
          };

          speechSynthesis.speak(utterance);
        };

        stop.onclick = function() {
          speechSynthesis.cancel();
          panel.classList.remove("speaking");
          status.textContent = "تم إيقاف الصوت";
        };
      </script>
    </body>
    </html>
    """

    voice_html = (
        voice_html
        .replace("__IMAGE__", avatar_b64)
        .replace("__TEXT__", json.dumps(
            latest_answer, ensure_ascii=False
        ))
        .replace(
            "__DISABLED__",
            "" if latest_answer else "disabled"
        )
    )

    with voice_slot:
        components.html(
            voice_html,
            height=390,
            scrolling=False
        )

# ===== END VOICE + LIGHT =====

st.caption(
    "SaeedLM نموذج Bigram تجريبي صغير. "
    "هذه الواجهة لا تعيد تدريب النموذج."
)
