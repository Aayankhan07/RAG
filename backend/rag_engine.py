import os
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from sentence_transformers import CrossEncoder

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Global lazy-loaded resources
_embeddings = None
_vectorstore = None
_reranker = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
        )
    return _vectorstore

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        # Load local cross-encoder for CPU execution
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker

def get_llm(model_name: str, temperature: float, api_keys: Dict[str, str]) -> Any:
    """Instantiate the appropriate LLM depending on the requested model name."""
    if model_name.startswith("gemini"):
        gemini_key = api_keys.get("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not configured in environment variables.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=gemini_key
        )
    else:
        groq_key = api_keys.get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY is not configured in environment variables.")
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=groq_key
        )

def process_pdf(pdf_path: str, filename: str) -> int:
    """Ingest a PDF using Semantic Chunking and add it to the Chroma vectorstore."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Configure semantic chunker with sentence-transformers embeddings
    text_splitter = SemanticChunker(
        get_embeddings(),
        breakpoint_threshold_type="percentile"
    )
    splits = text_splitter.split_documents(documents)

    # Force metadata source to be the original uploaded filename
    for split in splits:
        split.metadata["source"] = filename

    vectorstore = get_vectorstore()
    vectorstore.add_documents(splits)
    return len(splits)

def get_all_documents() -> List[Document]:
    """Retrieve all document chunks stored in Chroma."""
    vectorstore = get_vectorstore()
    db_data = vectorstore.get(include=["documents", "metadatas"])
    
    docs = []
    if db_data and "documents" in db_data:
        for i in range(len(db_data["documents"])):
            metadata = db_data["metadatas"][i] if db_data["metadatas"] else {}
            # Ensure metadata source has no temp folder path (basename only)
            if "source" in metadata:
                metadata["source"] = os.path.basename(metadata["source"])
            docs.append(Document(
                page_content=db_data["documents"][i],
                metadata=metadata
            ))
    return docs

def delete_document_by_source(source_name: str) -> bool:
    """Delete all chunks belonging to a specific source from Chroma."""
    vectorstore = get_vectorstore()
    records = vectorstore.get(where={"source": source_name})
    if records and records.get("ids"):
        vectorstore.delete(ids=records["ids"])
        return True
    return False

def get_hybrid_retriever(k: int) -> EnsembleRetriever:
    """
    Construct a Hybrid Ensemble Retriever:
    Dense Vector Search (Chroma) + Sparse Keyword Search (BM25) combined using RRF.
    """
    vectorstore = get_vectorstore()
    all_docs = get_all_documents()

    # Dense Retriever
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k * 2})

    if not all_docs:
        # Fall back to dense-only if database is empty
        return dense_retriever

    # Sparse Retriever
    sparse_retriever = BM25Retriever.from_documents(all_docs)
    sparse_retriever.k = k * 2

    # Ensemble: equal weight RRF
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def retrieve_and_rerank(query: str, k: int) -> List[Document]:
    """Retrieve candidates using Hybrid search and rerank them using Cross-Encoder."""
    try:
        retriever = get_hybrid_retriever(k)
        candidates = retriever.invoke(query)
    except Exception:
        # Fallback to pure similarity search if ensemble setup fails
        vectorstore = get_vectorstore()
        candidates = vectorstore.similarity_search(query, k=k * 2)

    if not candidates:
        return []

    # De-duplicate candidates based on page_content
    unique_candidates = []
    seen_contents = set()
    for doc in candidates:
        if doc.page_content not in seen_contents:
            unique_candidates.append(doc)
            seen_contents.add(doc.page_content)

    # Reranking stage using CrossEncoder
    try:
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in unique_candidates]
        scores = reranker.predict(pairs)

        # Sort documents by scores in descending order
        scored_docs = sorted(zip(scores, unique_candidates), key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for score, doc in scored_docs[:k]]
        return reranked_docs
    except Exception:
        # If reranking fails (e.g. model failed to load), return top k candidates directly
        return unique_candidates[:k]
