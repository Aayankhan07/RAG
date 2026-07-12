# 🧠 Advanced RAG & Hybrid Vector Search Engine

A production-grade, high-precision Retrieval-Augmented Generation (RAG) orchestrator featuring multi-stage document ingestion, semantic chunking, dense-sparse hybrid vector search, and cross-encoder reranking.

---

## 🚀 Key Features

* **Hybrid Retrieval Architecture:** Merges dense semantic embeddings (ChromaDB with `sentence-transformers/all-MiniLM-L6-v2`) with sparse keyword queries (BM25) using Reciprocal Rank Fusion (RRF) to maximize recall and precision.
* **Re-ranking Engine:** Integrates a local Cross-Encoder reranking module (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to evaluate context relevance, filter out noise, and prioritize top matches, drastically reducing LLM hallucinations.
* **Semantic Chunking:** Features layout-aware semantic splitting using sentence embeddings to divide documents at natural semantic transitions rather than arbitrary character splits.
* **Multi-LLM Orchestrator:** Seamless support for both **Groq API** (Llama 3.3, Mixtral, Gemma) and **Gemini API** (Gemini 1.5 Pro/Flash).
* **Dual-Service Architecture:** 
  * **Backend (FastAPI):** Exposes REST endpoints for uploading, listing, deleting, and query streaming (via NDJSON).
  * **Frontend (Streamlit):** Serves as an interactive interface with real-time stream rendering, parameter controllers, and dynamic source/citation citations.
* **Docker Containerization:** Fully containerized setup with volume mounts that cache HuggingFace models locally to prevent redundant downloads.

---

## 🛠️ Technology Stack

* **Orchestration:** LangChain
* **API Framework:** FastAPI
* **Frontend UI:** Streamlit
* **Vector Store:** ChromaDB
* **Embeddings & Reranking:** sentence-transformers, Cross-Encoder
* **LLM Gateways:** Groq API, Gemini API
* **Deployment:** Docker & Docker Compose

---

## 📂 Project Structure

```text
├── backend/
│   ├── main.py          # FastAPI application exposing query, ingestion, & delete REST endpoints
│   └── rag_engine.py    # Core RAG engine logic (embeddings, retrieval, reranker, LLM routing)
├── app.py               # Streamlit application (Frontend Client)
├── ingest.py            # CLI script for bulk semantic document ingestion from data/ folder
├── run_app.bat          # Windows batch script for running FastAPI & Streamlit locally
├── requirements.txt     # Python dependency configuration
├── Dockerfile.backend   # Docker construction for the FastAPI service
├── Dockerfile.frontend  # Docker construction for the Streamlit service
└── docker-compose.yml   # Docker compose configuration for dual-service setup
```

---

## 🔧 Installation & Setup

### Prerequisites
1. Get a **Groq API Key** from [Groq Console](https://console.groq.com/keys).
2. Get a **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).
3. Copy `.env.example` to `.env` and fill in your API keys:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Option A: Run via Docker (Recommended)
This method spins up both the FastAPI backend and Streamlit frontend inside isolated containers.

1. Build and run the containers:
   ```bash
   docker-compose up --build
   ```
2. Access the applications:
   * **Streamlit UI:** `http://localhost:8501`
   * **FastAPI Docs (Swagger UI):** `http://localhost:8000/docs`

### Option B: Local Run (No Docker)
To run directly on your host machine:

1. Double-click or run the batch script in your terminal:
   ```powershell
   .\run_app.bat
   ```
   *This automatically sets up a Python virtual environment, installs dependencies, starts the FastAPI backend server in a minimized background window, and opens the Streamlit frontend UI in your browser.*

---

## 🧪 Advanced CLI Ingestion
If you have a batch of PDF documents that you want to ingest offline before starting the web interface:
1. Place your PDFs inside the `data/` directory.
2. Activate your virtual environment and run:
   ```bash
   python ingest.py
   ```
   *This loads the PDFs, splits them semantically, and builds/updates the Chroma vector index in `chroma_db/`.*
