"""Central settings for the LangGraph RAG project."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "2"))
KB_FILE = BASE_DIR / "data" / "knowledge_base.txt"


def load_documents() -> list[str]:
    """Read the knowledge base: one fact per line, '#' lines are comments."""
    lines = KB_FILE.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
