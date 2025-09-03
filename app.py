import re
import streamlit as st
import faiss
import pickle
import torch
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    T5TokenizerFast,
    T5ForConditionalGeneration,
)
from datasets import load_dataset

st.set_page_config(page_title="Emotion‑Guided RAG Chat", layout="wide")
BASE = Path(__file__).parent

def find_folder(root: Path, must_have: list[str]) -> str:
    for cand in [root] + [p for p in root.iterdir() if p.is_dir()]:
        if all((cand / fn).exists() for fn in must_have):
            return str(cand.resolve())
    raise FileNotFoundError(f"No folder under {root} contains {must_have}")

@st.cache_resource
def load_all():
    ge_dir = find_folder(BASE/"aac_model_ge", ["config.json", "vocab.txt"])
    tok_ge = BertTokenizerFast.from_pretrained(ge_dir, local_files_only=True)
    mdl_ge = BertForSequenceClassification.from_pretrained(ge_dir, local_files_only=True)
    labels = load_dataset("go_emotions")["train"].features["labels"].feature.names

    ed_dir = find_folder(BASE/"aac_model_ed", ["spiece.model", "model.safetensors"])
    tok_ed = T5TokenizerFast.from_pretrained(ed_dir, local_files_only=True)
    mdl_ed = T5ForConditionalGeneration.from_pretrained(ed_dir, local_files_only=True)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index    = faiss.read_index("rag_index.faiss")
    contexts = pickle.load(open("contexts.pkl","rb"))

    return tok_ge, mdl_ge, labels, tok_ed, mdl_ed, embedder, index, contexts

tok_ge, mdl_ge, ge_labels, tok_ed, mdl_ed, embedder, index, contexts = load_all()

FEW_SHOT = """\
### Example 1
User: I need cheering up.
Assistant (warm, concise): You’ve got this! Think of one small win today.

### Example 2
User: I’m overwhelmed at work.
Assistant (friendly, concise): Let’s break it into steps—what’s one thing you can tackle right now?
"""

def retrieve(query: str, k: int) -> list[str]:
    emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    _, ids = index.search(emb, k)
    return [contexts[i] for i in ids[0]]

def build_prompt(user: str, emo: str, docs: list[str], prof: dict) -> str:
    return "\n\n".join([
        "SYSTEM: You are an empathetic assistant specialized in comforting users.",
        "GUIDELINES:\n"
        "1. Do NOT quote or invent personal details.\n"
        "2. First line must empathize (e.g. “I’m sorry you’re feeling scared.”).\n"
        "3. Summarize the BACKGROUND in one sentence.\n"
        "4. Then offer supportive advice.\n"
        "5. Use ‘you’, not ‘I’ when referring to personal actions.\n"
        "6. Follow STYLE exactly.",
        "FEW‑SHOT:\n" + FEW_SHOT,
        "STYLE:\n"
        f"• Greeting: {prof['greeting']}\n"
        f"• Formality: {', '.join(prof['formality'])}\n"
        f"• Length: {', '.join(prof['sentence_length'])}\n"
        f"• Personality: {', '.join(prof['personality'])}",
        "BACKGROUND (reference only):\n" +
        "\n".join(f"[REF {i+1}] {d}" for i, d in enumerate(docs)),
        f"USER: {user}\nEMOTION: {emo}",
        "ASSISTANT:"
    ])


def generate_response(user, prof, top_k):
    enc_ge = tok_ge(user, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        emo = ge_labels[mdl_ge(**enc_ge).logits.argmax(dim=-1).item()]

    docs = retrieve(user, top_k)

    prompt = build_prompt(user, emo, docs, prof)
    enc_ed = tok_ed(prompt, return_tensors="pt", truncation=True, padding=True)

    for _ in range(3):
        with torch.no_grad():
            out = mdl_ed.generate(
                **enc_ed,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                max_length=150,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
            )
        resp = tok_ed.decode(out[0], skip_special_tokens=True).replace("_comma_", ", ")
        if not _blackpat.search(resp):
            return emo, resp, docs

    
    fallback = "I’m sorry, I’m having trouble forming a helpful response right now. “You’re not alone, and it’s okay to feel this way.”"
    return emo, fallback, docs

st.title("Emotion‑Guided RAG Chat")
user_text = st.text_area("Enter your message:")

det = None
if user_text.strip():
    enc = tok_ge(user_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        det = ge_labels[mdl_ge(**enc).logits.argmax(dim=-1).item()]

emotion = st.selectbox(
    "Detected Emotion (override if needed)",
    options=ge_labels,
    index=ge_labels.index(det) if det else 0
) if det else None

st.sidebar.header("Style Controls")
formality   = st.sidebar.multiselect("Formality", ["casual","formal","semi-formal"], default=["casual"])
sent_length = st.sidebar.multiselect("Length", ["short","medium","detailed"],         default=["short"])
personality = st.sidebar.multiselect("Personality traits", ["warm","friendly","humorous","concise","empathetic"], default=["warm"])
greeting    = st.sidebar.text_input("Greeting", "Hey there!")
top_k       = st.sidebar.slider("Retrieval: top‑k docs", 1, 10, 5)

profile = {
    "formality": formality,
    "sentence_length": sent_length,
    "personality": personality,
    "greeting": greeting,
}

if st.button("Generate Reply"):
    if not user_text.strip():
        st.warning("Please enter a message!")
    else:
        with st.spinner("Generating…"):
            emo, resp, docs = generate_response(user_text, profile, top_k)

        st.subheader("Final Emotion Used")
        st.write(f"**{emotion or emo}**")

        st.subheader("Assistant Response")
        st.write(resp)

        st.subheader("Retrieved Contexts (reference only)")
        for i, d in enumerate(docs, 1):
            st.markdown(f"**[REF {i}]** {d}")
