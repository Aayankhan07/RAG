import os
import sys
import tempfile

import streamlit as st


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
def get_chat_groq() -> ChatGroq:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
    )
    return llm


def build_context(chunks) -> str:
    return "\n\n".join(chunk.page_content for chunk in chunks)


def main() -> None:
    st.set_page_config(page_title="RAG Chat (Groq)", page_icon="🧠")
    st.title("RAG Chat over your PDFs (Groq)")

    if not os.path.isdir(CHROMA_DIR):
        st.error("Chroma database not found. Run ingest.py to create chroma_db first.")
        return

    try:
        vectorstore = load_vectorstore()
    except Exception as exc:
        st.error(f"Failed to load Chroma database: {exc}")
        return

    try:
        llm = get_chat_groq()
    except Exception as exc:
        st.error(f"Failed to initialize ChatGroq: {exc}")
        return

    with st.sidebar:
        st.header("Knowledge base")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

        if uploaded_file is not None:
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
                    st.success("PDF processed and added to the knowledge base.")
                except Exception as exc:
                    st.error(f"Failed to process uploaded PDF: {exc}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about your documents")

    if not user_input:
        return

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
