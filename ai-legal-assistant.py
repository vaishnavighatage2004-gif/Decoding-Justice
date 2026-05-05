# ai-legal-assistant.py
import hashlib
import os
import io
import json
import uuid
import re
from typing import List, Optional

import streamlit as st
from gtts import gTTS
from langdetect import detect

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from langchain.memory import ConversationBufferMemory
from groq import Groq
import sounddevice as sd
import wavio
import speech_recognition as sr
from PyPDF2 import PdfReader

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

import chromadb

# ---------------------------
# Environment & page setup
# ---------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

st.set_page_config(page_title="AI Legal Assistant", layout="wide", page_icon="⚖️")

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(password, hashed_text):
    return make_hash(password) == hashed_text

USER_CREDENTIALS = {
    "admin": make_hash("admin123"),
    "lawyer": make_hash("legal123")
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:

    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username in USER_CREDENTIALS and check_password(
            password,
            USER_CREDENTIALS[username]
        ):

            st.session_state.authenticated = True
            st.success("Login Successful")
            st.rerun()

        else:
            st.error("Invalid Username or Password")

    st.stop()
def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(password, hashed_text):
    return make_hash(password) == hashed_text

USER_CREDENTIALS = {
    "admin": make_hash("admin123"),
    "lawyer": make_hash("legal123")
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    groq_client = None
    st.warning(f"Groq client initialization failed: {e}")

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_resource
def load_legal_model(model_name: str = "law-ai/InLegalBERT"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as exc:
        st.error(f"Failed to load model {model_name}: {exc}")
        raise

def get_legal_model():
    return load_legal_model()

@st.cache_resource
def get_chroma_client(path: str = "./legal_db"):
    try:
        client = chromadb.PersistentClient(path=path)
        return client
    except Exception as e:
        st.warning(f"Persistent Chroma client failed: {e}. Falling back to in-memory client.")
        return chromadb.Client()

chroma_client = get_chroma_client("./legal_db")
collection = chroma_client.get_or_create_collection(name="legal_docs")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def embed_text(text: str) -> List[float]:
    tokenizer, model = get_legal_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    vec = torch.nn.functional.normalize(
        outputs.last_hidden_state[:,0,:],
        p=2,
        dim=1
    ).cpu().numpy().flatten().tolist()
    return vec

def add_doc_to_chroma(doc_text: str, doc_id: Optional[str] = None, metadata: Optional[dict] = None):
    try:
        emb = embed_text(doc_text)
    except Exception as e:
        st.warning(f"Embedding failed: {e}")
        return None
    _id = doc_id or str(np.random.randint(1_000_000, 9_000_000))
    try:
        collection.add(documents=[doc_text], embeddings=[emb], ids=[_id], metadatas=[metadata or {}])
    except Exception:
        try:
            collection.add(documents=[doc_text], embeddings=[emb], ids=[_id])
        except Exception as e:
            st.warning(f"Failed to index document in Chroma: {e}")
            return None
    return _id

def chroma_search(query: str, k: int = 3):
    try:
        q_emb = embed_text(query)
    except Exception as e:
        st.warning(f"Embedding failed: {e}")
        return []
    try:
        res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "distances", "metadatas"])
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        results = []
        for i, doc in enumerate(docs):
            results.append({
                "text": doc,
                "distance": dists[i] if i < len(dists) else None,
                "metadata": metas[i] if i < len(metas) else {}
            })
        return results
    except Exception as e:
        st.warning(f"Chroma query failed: {e}")
        return []

def extract_text_from_pdf_bytes(pdf_bytes: bytes, allow_ocr: bool = False) -> str:
    text_chunks = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for p in reader.pages:
            try:
                t = p.extract_text()
            except Exception:
                t = None
            if t:
                text_chunks.append(t)
    except Exception:
        pass
    full_text = "\n".join(text_chunks).strip()
    if full_text:
        return full_text
    if allow_ocr and OCR_AVAILABLE:
        try:
            images = convert_from_bytes(pdf_bytes)
            ocr_parts = [pytesseract.image_to_string(img) for img in images]
            return "\n".join(ocr_parts)
        except Exception as e:
            st.warning(f"OCR failed: {e}")
            return ""
    return ""

# ---------------------------
# Regex & Metadata
# ---------------------------
SECTION_PATTERN = re.compile(
    r'(?:Section\s+\d+[A-Za-z()]*|IPC\s+\d+|CrPC\s+\d+|Article\s+\d+)',
    re.IGNORECASE
)
FIR_PATTERN = re.compile(r'FIR\s*No\.?\s*[:\-]?\s*[\w\/\-]+', re.IGNORECASE)
PS_PATTERN = re.compile(r'Police Station\s*[:\-]?\s*[A-Za-z ]+', re.IGNORECASE)
DATE_PATTERN = re.compile(r'(\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}|\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b)', re.IGNORECASE)

def extract_sections(text: str) -> List[str]:
    found = SECTION_PATTERN.findall(text)
    normalized = []
    for f in found:
        if isinstance(f, tuple):
            for part in f:
                if part and len(part.strip()) > 1:
                    normalized.append(part.strip())
        elif f and len(f.strip()) > 1:
            normalized.append(f.strip())
    return list(dict.fromkeys(normalized))[:30]

def extract_basic_metadata(text: str):
    firs = [m.group(0) for m in FIR_PATTERN.finditer(text)]
    ps = [m.group(0) for m in PS_PATTERN.finditer(text)]
    dates = [m.group(0) for m in DATE_PATTERN.finditer(text)]
    return {"firs": firs, "police_stations": ps, "dates": dates}

# ---------------------------
# LLM / Groq helper
# ---------------------------
def ask_groq(prompt: str, model_name: str = "llama-3.3-70b-versatile", max_tokens: int = 1024, temperature: float = 0.2):
    if not groq_client:
        st.error("Groq client not configured. Set GROQ_API_KEY to enable LLM features.")
        return "ERROR: GROQ client not configured."
    try:
        resp = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling Groq: {e}")
        return f"ERROR: {e}"

# ---------------------------
# PDF / Chat prompts
# ---------------------------
def build_pdf_analysis_prompt(extracted_text: str):
    prompt = f'''
You are an expert Indian legal analyst. Analyze the following legal document and provide clearly labeled sections.

1) SUMMARY: short 6-10 sentence summary.
2) FACTS OF THE CASE: bullet points.
3) KEY ARGUMENTS: prosecution and defense points.
4) SECTIONS INVOLVED: list of legal sections (IPC/CrPC/Evidence/etc).
5) FINAL JUDGMENT: court decision, verdict, sentence (if present).
6) CHARGES: likely charges or sections applicable if FIR/complaint.
7) METADATA_JSON: JSON containing FIR number, police station, dates, court name (or "Not found").
8) NEXT_STEPS: 3-5 actionable suggestions for an advocate.

Document:
{extracted_text[:5000]}

Return output exactly under these labels:
SUMMARY:
FACTS OF THE CASE:
KEY ARGUMENTS:
SECTIONS INVOLVED:
FINAL JUDGMENT:
CHARGES:
METADATA_JSON:
NEXT_STEPS:
'''
    return prompt

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("⚖️ Decoding Justice")
tabs = st.tabs(["Home", "PDF Analyzer", "Chat (Follow-up)", "Case Search", "Judgment Compare"])

# ---------------------------
# Tab 0 - Home
# ---------------------------
with tabs[0]:
    st.header("Welcome")
    st.markdown("""
    This app analyses legal PDFs (FIR/judgments/complaints), runs RAG (Chroma) search across uploaded docs,
    supports voice input, follow-up chat with memory, and judgment comparison.
    """)
    #st.info("Set your GROQ_API_KEY in the environment to enable LLM features.")

# ---------------------------
# Tab 1 - PDF Analyzer
# ---------------------------
with tabs[1]:
    col_top_left, col_top_right = st.columns([1,2])
    with col_top_left:
        uploaded_file = st.file_uploader("Upload PDF (FIR / Judgement / Complaint)", type=["pdf"], accept_multiple_files=False)
        allow_ocr = st.checkbox("Allow OCR fallback for scanned PDFs", value=True)
        index_doc = st.checkbox("Index this document into Chroma", value=True)
        add_metadata_input = st.text_input("Optional metadata JSON", value="")
    with col_top_right:
        st.write("Tips:")
        st.write("- Text PDFs extract best. Scanned PDFs require OCR (Tesseract).")
        st.write("- Indexed documents help Case Search tab find similar cases.")

    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        with st.spinner("Extracting text..."):
            extracted_text = extract_text_from_pdf_bytes(pdf_bytes, allow_ocr and OCR_AVAILABLE)
        if not extracted_text:
            st.error("No text extracted. Enable OCR if needed.")
        else:
            st.success(f"Extracted ~{len(extracted_text.split())} words.")
            if st.checkbox("Show extracted text", value=False):
                st.text_area("Extracted text", extracted_text[:4000], height=300)

            sections_found = extract_sections(extracted_text)
            metadata = extract_basic_metadata(extracted_text)
            st.write("**Sections (heuristic):**", sections_found[:10])
            st.write("**Metadata (heuristic):**")
            st.json(metadata)

            doc_id = None
            if index_doc:
                meta = {}
                try:
                    meta = json.loads(add_metadata_input) if add_metadata_input.strip() else {}
                except Exception:
                    meta = {"note": add_metadata_input}
                doc_id = add_doc_to_chroma(extracted_text, metadata=meta)
                st.session_state["last_uploaded_text"] = extracted_text
                st.success(f"Document indexed into Chroma (id={doc_id})")

            # LLM analysis button
            if st.button("Analysis LLM about this document"):
                if extracted_text:
                    with st.spinner("Asking LLM to analyze..."):
                        prompt = build_pdf_analysis_prompt(extracted_text)
                        analysis = ask_groq(prompt, max_tokens=1400, temperature=0.2)
                    if analysis:
                        st.subheader("📄 Document Analysis")
                        st.text_area("LLM Output", analysis, height=500)

                       


# ---------------------------
# Tab 2 - Chat Follow-up
# ---------------------------

def speech_to_text_sd(duration=5):
    fs = 44100  # Sampling rate
    st.info(f"🎤 Listening for {duration} seconds...")

    # Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    wavio.write("temp.wav", recording, fs, sampwidth=2)

    # Convert audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile("temp.wav") as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        st.success("✅ Recognized!")
        return text
    except sr.UnknownValueError:
        return "❌ Could not understand audio."
    except sr.RequestError:
        return "⚠️ Could not request results; check your internet connection."

def speak_same_language(text, user_query_language='en'):
    """
    Convert text to speech and play in browser using Streamlit audio.
    """
    try:
        tts = gTTS(text=text, lang=user_query_language)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)  # Move pointer to the start
        st.audio(mp3_fp, format='audio/mp3')
    except Exception as e:
        st.error(f"TTS failed: {e}")

# -----------------------------
# Function: Build prompt for follow-up questions
# -----------------------------
def build_followup_prompt(context, question):
    return f"""
You are an AI Legal Assistant.

Context:
{context}

User Question:
{question}

Answer clearly and professionally.
"""


# -----------------------------
# Streamlit tab: Follow-up Chat
# -----------------------------
with tabs[2]:
    st.header("💬 Chat (Follow-up)")
    st.write("Ask questions about uploaded or indexed documents. Memory is per-session.")

    if st.button("Clear Conversation Memory"):
        memory.clear()
        st.success("Conversation memory cleared.")

    voice_col, text_col = st.columns(2)

    # --- Voice Input ---
    with voice_col:
        if st.button("🎤 Use Voice Input"):
            with st.spinner("Listening..."):
                spoken = speech_to_text_sd(duration=5)
            if spoken and not spoken.startswith(("❌","⚠️","⏰")):
                user_query = spoken
            else:
                user_query = ""
        else:
            user_query = ""

    # --- Text Input fallback ---
    with text_col:
        if not user_query:
            user_query = st.text_input("Type your question:")

    # --- Ask LLM ---
    if user_query:
        # Get last uploaded document or search in Chroma
        doc_excerpt = st.session_state.get("last_uploaded_text", "")
        if not doc_excerpt:
            top_chroma = chroma_search(user_query, k=1)
            doc_excerpt = top_chroma[0]["text"][:12000] if top_chroma else ""

        # Build follow-up prompt
        follow_prompt = build_followup_prompt(doc_excerpt or "", user_query)

        # Query LLM
        with st.spinner("Asking LLM..."):
            resp = ask_groq(follow_prompt, max_tokens=700, temperature=0.15)

        if resp:
            st.markdown("**Assistant:**")
            st.write(resp)
            speak_same_language(resp, user_query_language='en')  # or detect language dynamically




# ---------------------------
# Tab 3 - Case Search
# ---------------------------
with tabs[3]:
    st.header("🔎 Case Search")
    q = st.text_input("Enter search query / case excerpt:")
    top_k = st.slider("Number of results", 1, 10, 3)
    if st.button("Search"):
        if not q.strip():
            st.warning("Enter a query.")
        else:
            results = chroma_search(q, k=top_k)
            if not results:
                st.info("No indexed documents found.")
            else:
                for i, r in enumerate(results, start=1):
                    st.subheader(f"Result #{i} (distance={r.get('distance')})")
                    st.write(r.get("text")[:1500])
                    if r.get("metadata"):
                        st.write("Metadata:", r.get("metadata"))
                    #if st.button(f"Ask LLM about this result #{i}", key=f"btn_{i}"):
                        #text = r.get("text", "")

# ---------------------------
# Tab 4 - Judgment Comparison
# ---------------------------
# ---------------------------
# Tab 4 - Judgment Comparison
# ---------------------------
# ---------------------------
# Tab 4 - Judgment Comparison
# ---------------------------
with tabs[4]:
    st.header("⚖️ Judgment Comparison (A vs B)")
    st.markdown("Upload or paste two documents to compare.")

    colA, colB = st.columns(2)

    with colA:
        fileA = st.file_uploader("Upload PDF A", type=["pdf"], key="fileA")
        pasteA = st.text_area("Or paste text A", height=150, key="pasteA")

    with colB:
        fileB = st.file_uploader("Upload PDF B", type=["pdf"], key="fileB")
        pasteB = st.text_area("Or paste text B", height=150, key="pasteB")

    textA = ""
    textB = ""

    # ---- Load Document A ----
    if fileA:
        bytesA = fileA.read()
        textA = extract_text_from_pdf_bytes(bytesA, allow_ocr=False)
    elif pasteA.strip():
        textA = pasteA.strip()

    # ---- Load Document B ----
    if fileB:
        bytesB = fileB.read()
        textB = extract_text_from_pdf_bytes(bytesB, allow_ocr=False)
    elif pasteB.strip():
        textB = pasteB.strip()

    # ---- Compare Button ----
    if st.button("Compare A vs B"):
        if not textA or not textB:
            st.error("Please provide both documents.")
        else:
            # Limit size BEFORE sending to LLM
            textA = textA[:6000]
            textB = textB[:6000]

            with st.spinner("Comparing documents..."):
                prompt = build_compare_prompt(textA, textB)
                comparison = ask_groq(prompt, max_tokens=1200, temperature=0.2)

            if comparison:
                st.subheader("Comparison Result (LLM):")
                st.write(comparison)

                # Optional quick summaries
                sA = ask_groq(
                    f"Give a 3-sentence summary of document A:\n\n{textA}",
                    max_tokens=300,
                    temperature=0.15
                )

                sB = ask_groq(
                    f"Give a 3-sentence summary of document B:\n\n{textB}",
                    max_tokens=300,
                    temperature=0.15
                )

                st.write("### Quick Summaries")
                st.write("**Document A:**", sA)
                st.write("**Document B:**", sB)