import os
import sys
import json
import requests
import streamlit as st
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Setup sys.path for local packages folder
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_PACKAGES_DIR = os.path.join(PROJECT_ROOT, ".packages")
if os.path.isdir(PROJECT_PACKAGES_DIR) and PROJECT_PACKAGES_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PACKAGES_DIR)

# Default backend URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

def trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Import premium modern font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        /* Clean titles & headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }

        /* Custom source container card */
        .source-card {
            background: rgba(128, 128, 128, 0.05);
            border-left: 4px solid #6366F1;
            border-radius: 0 10px 10px 0;
            padding: 12px 16px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            transition: transform 0.2s ease;
        }
        .source-card:hover {
            transform: translateX(2px);
            background: rgba(128, 128, 128, 0.08);
        }
        
        .source-badge {
            background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
            color: white !important;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
            box-shadow: 0 2px 5px rgba(79, 70, 229, 0.2);
        }

        /* Sidebar Styling overrides */
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
            border-right: 1px solid rgba(255,255,255,0.06) !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #F8FAFC !important;
        }
        
        /* Premium custom styling for sidebar buttons */
        div[data-testid="stSidebar"] button {
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            background-color: rgba(255,255,255,0.03) !important;
            color: #E2E8F0 !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stSidebar"] button:hover {
            background-color: rgba(255,255,255,0.08) !important;
            border-color: #6366F1 !important;
            color: white !important;
            box-shadow: 0 0 8px rgba(99, 102, 241, 0.2);
        }

        /* Sleek card container for files */
        .file-card {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
        }
        
        /* Adjust chat messages padding and shadow */
        .stChatMessage {
            border-radius: 12px !important;
            padding: 1.2rem 1.6rem !important;
            margin-bottom: 1.2rem !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.01) !important;
            border: 1px solid rgba(128,128,128,0.06) !important;
        }
        
        /* Streamlit expander custom CSS */
        .stExpander {
            border-radius: 10px !important;
            border: 1px solid rgba(128,128,128,0.12) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.01) !important;
            overflow: hidden;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

def display_sources(sources: list):
    if not sources:
        return
    with st.expander("🔍 View Retrieved Sources"):
        for idx, src in enumerate(sources, 1):
            source_name = src.get("source", "Unknown")
            page_num = src.get("page", 1)
            content = src.get("content", "")
            st.markdown(
                f"""
                <div class="source-card">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span class="source-badge">Source {idx}</span>
                        <span style="font-size: 0.9rem; font-weight: 600;">📄 {source_name}</span>
                        <span style="margin-left: auto; font-size: 0.8rem; opacity: 0.6; font-weight: 500;">Page {page_num}</span>
                    </div>
                    <div style="font-size: 0.88rem; line-height: 1.6; opacity: 0.9; padding-left: 2px;">
                        {content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def get_documents_list() -> list:
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json().get("documents", [])
    except Exception:
        pass
    return []

def upload_pdf(file_bytes, filename: str) -> tuple:
    try:
        files = {"file": (filename, file_bytes, "application/pdf")}
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
        if response.status_code == 200:
            return True, f"Successfully uploaded and indexed '{filename}'"
        else:
            detail = response.json().get("detail", "Error processing file.")
            return False, f"Failed: {detail}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def delete_pdf(filename: str) -> tuple:
    try:
        response = requests.delete(f"{BACKEND_URL}/documents/{filename}", timeout=10)
        if response.status_code == 200:
            return True, f"Successfully deleted '{filename}'"
        else:
            detail = response.json().get("detail", "Error deleting file.")
            return False, f"Failed: {detail}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def query_stream(query_text: str, model: str, temp: float, k: int, history_messages: list):
    payload = {
        "query": query_text,
        "model_name": model,
        "temperature": temp,
        "k": k,
        "history": []
    }
    
    try:
        response = requests.post(f"{BACKEND_URL}/query", json=payload, stream=True)
        if response.status_code != 200:
            yield {"type": "error", "data": f"Server error: {response.text}"}
            return
            
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                try:
                    yield json.loads(decoded)
                except json.JSONDecodeError:
                    yield {"type": "error", "data": f"Failed to parse backend data: {decoded}"}
    except Exception as e:
        yield {"type": "error", "data": f"Failed to communicate with backend: {str(e)}"}

def main() -> None:
    st.set_page_config(page_title="Advanced RAG Engine", page_icon="🧠", layout="wide")
    inject_custom_css()
    
    # Premium Header Block
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 25px;">
            <div style="background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%); padding: 12px; border-radius: 12px; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);">
                <span style="font-size: 2rem; display: block; line-height: 1;">🧠</span>
            </div>
            <div>
                <h1 style="margin: 0; padding: 0; line-height: 1.1; font-size: 2.2rem; font-weight: 700; color: inherit;">Advanced RAG Orchestrator</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.7; font-size: 1rem; font-weight: 500;">Dense-Sparse Hybrid Retrieval (BM25 + Chroma) & Cross-Encoder Reranking</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize messages list
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    with st.sidebar:
        st.header("Settings & Tools")
        
        # 1. Clear Chat History
        if st.button("🗑️ Clear Chat History"):
            st.session_state["messages"] = []
            st.success("Chat history cleared!")

        st.divider()

        # 2. LLM Configuration (Multi-API)
        st.header("LLM Orchestrator")
        model_options = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        selected_model = st.selectbox("Select LLM Model", options=model_options, index=0)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
        k_chunks = st.slider("Retrieve Chunks (k)", min_value=1, max_value=10, value=3, step=1)

        st.divider()

        # 3. Knowledge Base List & Delete Management
        st.header("Knowledge Base")
        
        # Display backend connection health check
        backend_healthy = False
        try:
            health_response = requests.get(f"{BACKEND_URL}/documents", timeout=2)
            if health_response.status_code == 200:
                backend_healthy = True
        except Exception:
            pass

        if not backend_healthy:
            st.error(f"🔴 Connection to backend server ({BACKEND_URL}) failed.")
            st.info(
                "**First-time startup note:** The backend downloads the required machine learning models "
                "(embeddings & rerankers) on its first boot. This download can take a minute. "
                "Please wait a moment and refresh this page."
            )
            if st.button("🔄 Check Connection Again"):
                trigger_rerun()
            return

        # Fetch and list document names
        unique_sources = get_documents_list()
        
        if unique_sources:
            st.subheader("Managed Documents")
            for source in unique_sources:
                col1, col2 = st.columns([0.85, 0.15])
                col1.markdown(f"<div style='padding-top: 4px; font-weight: 500; font-size: 0.9rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #E2E8F0;'>📄 {source}</div>", unsafe_allow_html=True)
                if col2.button("🗑️", key=f"del_{source}", help=f"Delete {source}"):
                    success, msg = delete_pdf(source)
                    if success:
                        st.success(msg)
                        trigger_rerun()
                    else:
                        st.error(msg)

        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"])

        if uploaded_file is not None:
            # We index files and immediately update the list
            if "processed_files" not in st.session_state:
                st.session_state["processed_files"] = set()

            if uploaded_file.name not in st.session_state["processed_files"]:
                with st.spinner("Uploading and running semantic ingestion..."):
                    success, msg = upload_pdf(uploaded_file.getvalue(), uploaded_file.name)
                    if success:
                        st.session_state["processed_files"].add(uploaded_file.name)
                        st.success(msg)
                        trigger_rerun()
                    else:
                        st.error(msg)
            else:
                st.info(f"'{uploaded_file.name}' is already loaded in this session.")

        st.divider()
        st.header("Database Status")
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

    # Render existing conversation history (with source citation expanders)
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                display_sources(message["sources"])

    # User chat query input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        st.session_state["messages"] = [{"role": "user", "content": user_input}]

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Searching documents, running Hybrid Retrieval and Reranker...")
            
            # Setup placeholder containers for streaming
            sources_info = []
            response_text = ""
            
            # Perform query streaming from FastAPI Backend
            for packet in query_stream(
                query_text=user_input, 
                model=selected_model, 
                temp=temperature, 
                k=k_chunks, 
                history_messages=st.session_state["messages"][:-1]
            ):
                packet_type = packet.get("type")
                packet_data = packet.get("data")
                
                if packet_type == "sources":
                    sources_info = packet_data
                elif packet_type == "token":
                    response_text += packet_data
                    placeholder.markdown(response_text + "▌")
                elif packet_type == "error":
                    st.error(packet_data)
                    return
            
            # Final output presentation without cursor
            placeholder.markdown(response_text)
            
            if sources_info:
                display_sources(sources_info)
                        
            # Store generation metadata in state
            st.session_state["messages"].append({
                "role": "assistant",
                "content": response_text,
                "sources": sources_info
            })

if __name__ == "__main__":
    main()
