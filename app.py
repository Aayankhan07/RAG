import os
import sys
import tempfile

import streamlit as st
import psycopg2

from dotenv import load_dotenv
load_dotenv()

PROJECT_PACKAGES_DIR = os.path.join(os.path.dirname(__file__), ".packages")

if os.path.isdir(PROJECT_PACKAGES_DIR) and PROJECT_PACKAGES_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PACKAGES_DIR)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROMPT_TEMPLATE = (
    "Use the following context to answer the user's question. "
    "If you don't know the answer, just say you don't know. "
    "Context: {context}. Question: {question}"
)


@st.cache_resource
def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


@st.cache_resource
def get_chat_groq(api_key: str) -> ChatGroq:
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    return llm


def build_context(chunks) -> str:
    return "\n\n".join(chunk.page_content for chunk in chunks)


def main() -> None:
    st.set_page_config(page_title="RAG Chat (Groq)", page_icon="🧠")
    st.title("RAG Chat over your PDFs (Groq)")

    try:
        vectorstore = load_vectorstore()
    except Exception as exc:
        st.error(f"Failed to load Chroma database: {exc}")
        return

    # Initialize session state for messages early so the clear button can use it
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with st.sidebar:
        st.header("Settings & Tools")
        
        # 1. The Clear Chat Button
        if st.button("🗑️ Clear Chat History"):
            st.session_state["messages"] = []
            st.success("Chat history cleared!")

        st.divider() 
        
        # 2. PDF Uploader (API key text box removed entirely)
        st.header("Knowledge Base")
        uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"])

        # Track processed files so we don't re-index on every app interaction
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = set()

        if uploaded_file is not None:
            # Check if we have already indexed this exact file
            if uploaded_file.name not in st.session_state["processed_files"]:
                with st.spinner("Reading and indexing your PDF..."):
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_path = tmp_file.name

                        loader = PyPDFLoader(tmp_path)
                        documents = loader.load()

                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50,
                        )
                        splits = splitter.split_documents(documents)

                        vectorstore.add_documents(splits)
                        
                        # Mark this file as processed in the session state
                        st.session_state["processed_files"].add(uploaded_file.name)
                        st.success("PDF processed and added to the knowledge base.")
                    except Exception as exc:
                        st.error(f"Failed to process uploaded PDF: {exc}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
            else:
                # If already processed, just show a persistent success message
                st.success(f"'{uploaded_file.name}' is loaded and ready.")
        
        st.header("Database")
        db_status = "Not configured"
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            try:
                conn = psycopg2.connect(dsn=db_url, connect_timeout=5)
                conn.close()
                db_status = "Connected"
            except Exception as exc:
                db_status = f"Error: {exc}"
        st.write(f"Postgres: {db_status}")

    # Pull the key invisibly from your .env file
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        st.error("Groq API Key is missing. Please add it to your .env file.")
        return
        
    try:
        llm = get_chat_groq(api_key)
    except Exception as exc:
        st.error(f"Failed to initialize ChatGroq: {exc}")
        return

    # Render existing chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about your documents")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Searching your documents and thinking...")

            try:
                docs = vectorstore.similarity_search(user_input, k=3)
            except Exception as exc:
                placeholder.markdown(f"Error during similarity search: {exc}")
                return

            if not docs:
                answer = "I could not find any relevant context in the knowledge base."
                placeholder.markdown(answer)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer}
                )
                return

            context = build_context(docs)
            filled_prompt = PROMPT_TEMPLATE.format(context=context, question=user_input)

            try:
                response = llm.invoke(filled_prompt)
                answer = response.content
            except Exception as exc:
                answer = f"Error from Groq LLM: {exc}"

            placeholder.markdown(answer)
            st.session_state["messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
