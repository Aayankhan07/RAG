import os
import sys
import json
import tempfile
from typing import List

# Setup sys.path for local packages folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_PACKAGES_DIR = os.path.join(PROJECT_ROOT, ".packages")
if os.path.isdir(PROJECT_PACKAGES_DIR) and PROJECT_PACKAGES_DIR not in sys.path:
    sys.path.insert(0, PROJECT_PACKAGES_DIR)

# Ensure project root is in sys.path so we can import 'backend'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from backend import rag_engine

app = FastAPI(title="Advanced Hybrid RAG API")

# Enable CORS for communication from Streamlit client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class QueryPayload(BaseModel):
    query: str
    model_name: str
    temperature: float = 0.3
    k: int = 3
    history: List[Message] = []

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, chunk it semantically, and index it into vector and sparse databases."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        try:
            chunks_created = rag_engine.process_pdf(tmp_path, file.filename)
            return {
                "status": "success",
                "filename": file.filename,
                "chunks_created": chunks_created
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.get("/documents")
async def get_documents():
    """Get a list of unique source document names currently stored in the system."""
    try:
        docs = rag_engine.get_all_documents()
        unique_sources = set()
        for doc in docs:
            source = doc.metadata.get("source")
            if source:
                unique_sources.add(os.path.basename(source))
        return {"documents": sorted(list(unique_sources))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{name}")
async def delete_document(name: str):
    """Remove a source document and all of its chunks from the database."""
    try:
        success = rag_engine.delete_document_by_source(name)
        if success:
            return {"status": "success", "message": f"Deleted document '{name}'."}
        else:
            raise HTTPException(status_code=404, detail=f"Document '{name}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(payload: QueryPayload):
    """Perform hybrid search, Cross-Encoder reranking, and stream the LLM response along with retrieved sources."""
    # 1. Retrieve and Rerank chunks
    docs = rag_engine.retrieve_and_rerank(payload.query, payload.k)
    
    # 2. Extract source attribution metadata
    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1,
            "content": doc.page_content
        })
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Format conversational messages
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    system_prompt = (
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer, just say you don't know. Do not make up facts.\n\n"
        f"Context:\n{context}"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=payload.query)
    ]
    
    # 4. Initialize dynamic LLM (Groq / Gemini)
    api_keys = {
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", ""),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "")
    }
    
    try:
        llm = rag_engine.get_llm(payload.model_name, payload.temperature, api_keys)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    
    # 5. Generator function to stream NDJSON packets
    def response_streamer():
        # First send retrieved sources information
        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        
        # Stream LLM generation tokens
        try:
            for chunk in llm.stream(messages):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                yield json.dumps({"type": "token", "data": content}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": f"Generation error: {str(e)}"}) + "\n"
            
    return StreamingResponse(response_streamer(), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
