import sys
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path so `from src.rag_pipeline import ...` resolves correctly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_pipeline import RagPipeline

app = FastAPI(
    title="LearnRAG API",
    description="API for the LearnRAG AI Engineering Knowledge & Troubleshooting System"
)

# Global pipeline instance to avoid reloading models on every request
_pipeline = None

def get_pipeline(fetch_k: int = 12, top_k: int = 5) -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(fetch_k=fetch_k, top_k=top_k)
    else:
        # Dynamically update the parameters if the user requested different ones
        # Note: Depending on your RerankingRetriever implementation, you might need to set these on the internal retrievers.
        _pipeline.retriever.fetch_k = fetch_k
        _pipeline.retriever.top_k = top_k
    return _pipeline

class QueryRequest(BaseModel):
    query: str
    fetch_k: int = 12
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    context: List[str]
    latency: float

@app.post("/api/chat", response_model=QueryResponse)
def process_query(request: QueryRequest):
    try:
        rag = get_pipeline(fetch_k=request.fetch_k, top_k=request.top_k)
        
        started = time.perf_counter()
        result = rag.invoke(request.query)
        latency = time.perf_counter() - started
        
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            context=result["context"],
            latency=latency
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
def health_check():
    return {"status": "ok"}