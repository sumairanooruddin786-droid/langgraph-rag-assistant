# LangGraph RAG Assistant

A small web app that answers questions using **Retrieval-Augmented Generation (RAG)**.
A LangGraph workflow first retrieves the most relevant facts from a knowledge base, then a
Groq-hosted language model writes an answer using only those facts. The web page also shows
the retrieved facts, so you can see where the answer came from.

## How it works

```
Question → [retrieve] → [generate] → Answer + sources
```

| Part | What it does |
|------|--------------|
| **State** | Shared data between steps: `question`, `context`, `answer` |
| **`retrieve` node** | Embeds the question and finds the top 2 similar facts (`InMemoryVectorStore`) |
| **`generate` node** | Sends the facts and question to Groq (`openai/gpt-oss-120b`); the model must answer only from the facts |
| **Edges** | `START → retrieve → generate → END` |

## Tech stack

- **LangGraph** for the workflow (state, nodes, edges)
- **Groq** via `langchain-groq` for the chat model
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for embeddings, running on CPU
- **FastAPI** + **Uvicorn** for the backend
- **HTML, CSS and JavaScript** for the frontend (no build step)

## Folder structure

```
lang_graph/
├── app/
│   ├── __init__.py
│   ├── config.py          # settings, knowledge base loader
│   ├── graph.py           # the LangGraph workflow
│   └── main.py            # FastAPI server and API routes
├── frontend/
│   └── index.html         # single-page user interface
├── data/
│   └── knowledge_base.txt # one fact per line
├── docs/
│   ├── LangGraph_RAG_Presentation.pptx
│   └── Project_Report     # report (add your exported copy here)
├── legacy/
│   └── langgraph_practice.py   # original class script (command line)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

You need **Python 3.10 or newer** and a free **Groq API key** from <https://console.groq.com>.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install packages
pip install -r requirements.txt

# 3. Add your API key
copy .env.example .env         # Windows  (use cp on macOS / Linux)
# then open .env and paste your key after GROQ_API_KEY=
```

## Run

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000> in your browser and ask a question such as
*"What is a node in LangGraph?"*

The first question takes longer because the embedding model loads once.

To run the original command-line version instead:

```bash
python legacy/langgraph_practice.py
```

## API

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/health` | Shows model names, whether the key is set, and how many facts are loaded |
| `POST` | `/api/ask` | Body: `{"question": "..."}`. Returns `question`, `answer`, `sources` |

Interactive API docs are available at <http://127.0.0.1:8000/docs>.

## Customize

- **Add knowledge:** write new facts as lines in `data/knowledge_base.txt`, then restart the server.
- **Change settings:** set `GROQ_MODEL` or `TOP_K` in `.env`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY is missing` | Create `.env` from `.env.example` and paste your key |
| First request is slow | Normal: the embedding model downloads and loads once |
| `ModuleNotFoundError` | Activate the virtual environment, then run `pip install -r requirements.txt` |
| Model name rejected | Check the current model list on the Groq console and set `GROQ_MODEL` |

## Notes for submission

Do **not** include the `venv/` folder when you zip or upload the project. It is very large and
anyone can recreate it with `pip install -r requirements.txt`. The `.gitignore` already excludes it.

## Future work

- Add a conditional edge that asks the user to rephrase when no facts match
- Load PDFs or notes into the knowledge base
- Trace runs and evaluate answers with LangSmith
