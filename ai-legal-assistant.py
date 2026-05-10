# ai-legal-assistant.py

import os
import json
import streamlit as st
from langchain.memory import ConversationBufferMemory

from utils.auth import (
    register_user,
    login_user
)

from utils.pdf_utils import (
    extract_text_from_pdf_bytes,
    extract_sections,
    extract_basic_metadata,
    OCR_AVAILABLE
)

from utils.rag_utils import (
    add_doc_to_chroma,
    chroma_search
)

from utils.llm_utils import (
    ask_groq
)

from utils.prompts import (
    build_pdf_analysis_prompt,
    build_followup_prompt,
    build_compare_prompt
)

from utils.translate_utils import (
    detect_language,
    translate_to_english,
    translate_from_english
)

# ---------------------------
# Streamlit Config
# ---------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

st.set_page_config(
    page_title="AI Legal Assistant",
    layout="wide",
    page_icon="⚖️"
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "last_uploaded_text" not in st.session_state:
    st.session_state.last_uploaded_text = ""

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


if not st.session_state.authenticated:

    st.title("🔐 Login / Register")

    menu = ["Login", "Register"]

    choice = st.sidebar.selectbox(
        "Menu",
        menu
    )

    
    if choice == "Register":

        new_user = st.text_input(
            "Create Username"
        )

        new_password = st.text_input(
            "Create Password",
            type="password"
        )

        if st.button("Register"):

            try:
                register_user(
                    new_user,
                    new_password
                )

                st.success(
                    "User Registered Successfully"
                )

            except:
                st.error(
                    "Username already exists"
                )


    elif choice == "Login":

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            result = login_user(
                username,
                password
            )

            if result:

                st.session_state.authenticated = True

                st.success(
                    "Login Successful"
                )

                st.rerun()

            else:

                st.error(
                    "Invalid Username or Password"
                )

    st.stop()


st.title("⚖️ Decoding Justice")
language_options = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

selected_language = st.sidebar.selectbox(
    "Select Language",
    list(language_options.keys())
)

tabs = st.tabs([
    "Home",
    "PDF Analyzer",
    "Chat",
    "Case Search",
    "Judgment Compare"
])


with tabs[0]:

    st.header("🏠 Welcome")

    st.markdown("""
                
                “Decoding Justice” is an AI-based legal assistant system developed to help advocates, students, and legal professionals analyze legal documents efficiently.
                 The project reduces manual effort in reading lengthy legal documents such as FIRs, judgments, complaints, and case files.
    ### Features

    - Legal PDF Analysis
    - OCR for Scanned PDFs
    - ChromaDB Semantic Search
    - Legal AI Chat
    - Judgment Comparison
    - RAG-based Retrieval
    """)


with tabs[1]:

    st.header("📄 PDF Analyzer")

    uploaded_file = st.file_uploader(
        "Upload Legal PDF",
        type=["pdf"]
    )

    allow_ocr = st.checkbox(
        "Enable OCR",
        value=True
    )

    index_doc = st.checkbox(
        "Index Document into ChromaDB",
        value=True
    )

    metadata_input = st.text_input(
        "Optional Metadata JSON"
    )

    if uploaded_file:

        pdf_bytes = uploaded_file.read()

        with st.spinner("Extracting text..."):

            extracted_text = extract_text_from_pdf_bytes(
                pdf_bytes,
                allow_ocr and OCR_AVAILABLE
            )

        if not extracted_text:

            st.error(
                "No text extracted from PDF"
            )

        else:

            st.success(
                f"Extracted {len(extracted_text.split())} words"
            )

            if st.checkbox("Show Extracted Text"):

                st.text_area(
                    "Extracted Text",
                    extracted_text[:4000],
                    height=300
                )

           
            sections = extract_sections(
                extracted_text
            )

            st.write("### Sections Found")
            st.write(sections)

            
            metadata = extract_basic_metadata(
                extracted_text
            )

            st.write("### Metadata")
            st.json(metadata)

            
            st.session_state.last_uploaded_text = (
                extracted_text
            )

            
            if index_doc:

                try:

                    meta = (
                        json.loads(metadata_input)
                        if metadata_input.strip()
                        else {}
                    )

                except:

                    meta = {
                        "note": metadata_input
                    }

                add_doc_to_chroma(
                    extracted_text
                )

                st.success(
                    "Document indexed successfully"
                )

        
            if st.button("Analyze Document"):

                with st.spinner(
                    "Analyzing with AI..."
                ):

                    prompt = build_pdf_analysis_prompt(
                        extracted_text
                    )

                    analysis = ask_groq(
                        prompt,
                        max_tokens=1400
                    )

                st.subheader(
                    "📘 AI Analysis"
                )

                st.write(analysis)

with tabs[2]:

    st.header("💬 Chat (Follow-up)")

    user_query = st.text_input(
        "Type your question:"
    )

    if user_query:

        doc_excerpt = st.session_state.get(
            "last_uploaded_text",
            ""
        )

        if not doc_excerpt:

            top_chroma = chroma_search(
                user_query,
                k=1
            )

            if top_chroma:
                doc_excerpt = top_chroma[0]["text"][:12000]

            else:
                doc_excerpt = ""

        follow_prompt = build_followup_prompt(
            doc_excerpt,
            user_query
        )

        with st.spinner("Asking LLM..."):

            resp = ask_groq(
                follow_prompt,
                max_tokens=700,
                temperature=0.15
            )

            translated_resp = translate_from_english(
                resp,
                language_options[selected_language]
            )

        st.markdown("**Assistant:**")
        st.write(translated_resp)

    
with tabs[3]:

    st.header("🔎 Case Search")

    search_query = st.text_input(
        "Enter Search Query"
    )

    top_k = st.slider(
        "Number of Results",
        1,
        10,
        3
    )

    if st.button("Search Cases"):

        results = chroma_search(
            search_query,
            k=top_k
        )

        docs = results.get(
            "documents",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        if not docs:

            st.warning(
                "No Results Found"
            )

        else:

            for i, doc in enumerate(docs):

                st.subheader(
                    f"Result {i+1}"
                )

                if i < len(distances):

                    st.write(
                        f"Distance: {distances[i]}"
                    )

                st.write(doc[:1500])


with tabs[4]:

    st.header("⚖️ Judgment Comparison")

    col1, col2 = st.columns(2)

    with col1:

        fileA = st.file_uploader(
            "Upload PDF A",
            type=["pdf"],
            key="A"
        )

        pasteA = st.text_area(
            "Or Paste Text A"
        )

    with col2:

        fileB = st.file_uploader(
            "Upload PDF B",
            type=["pdf"],
            key="B"
        )

        pasteB = st.text_area(
            "Or Paste Text B"
        )

    textA = ""
    textB = ""

    if fileA:

        textA = extract_text_from_pdf_bytes(
            fileA.read(),
            allow_ocr=False
        )

    elif pasteA.strip():

        textA = pasteA

   
    if fileB:

        textB = extract_text_from_pdf_bytes(
            fileB.read(),
            allow_ocr=False
        )

    elif pasteB.strip():

        textB = pasteB

    if st.button("Compare Documents"):

        if not textA or not textB:

            st.error(
                "Please provide both documents"
            )

        else:

            prompt = build_compare_prompt(
                textA[:6000],
                textB[:6000]
            )

            with st.spinner(
                "Comparing..."
            ):

                comparison = ask_groq(
                    prompt,
                    max_tokens=1200
                )

            st.subheader(
                "Comparison Result"
            )

            st.write(comparison)