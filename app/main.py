"""FastAPI server: serves the frontend and exposes the LangGraph workflow."""
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import BASE_DIR, EMBED_MODEL, GROQ_MODEL, load_documents

FRONTEND = BASE_DIR / "frontend"
app = FastAPI(title="LangGraph RAG Assistant")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


@lru_cache(maxsize=1)
def get_graph():
    """Build the graph once (first request) and reuse it."""
    from .graph import build_graph
    return build_graph()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
        "model": GROQ_MODEL,
        "embedding_model": EMBED_MODEL,
        "documents": len(load_documents()),
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(503, "GROQ_API_KEY is missing. Add it to the .env file.")
    try:
        result = get_graph().invoke(
            {"question": req.question, "context": "", "answer": ""}
        )
    except Exception as exc:  # surface model/network problems to the UI
        raise HTTPException(500, f"Workflow failed: {exc}")
    sources = [s for s in result["context"].split("\n") if s]
    return {"question": req.question, "answer": result["answer"], "sources": sources}


app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
