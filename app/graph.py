"""LangGraph workflow: retrieve relevant facts, then generate an answer."""
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .config import EMBED_MODEL, GROQ_MODEL, TOP_K, load_documents


class State(TypedDict):
    """Data shared between the nodes."""
    question: str
    context: str
    answer: str


def build_graph():
    """Create the embeddings, vector store, model and compiled graph."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}
    )
    store = InMemoryVectorStore.from_texts(load_documents(), embedding=embeddings)
    model = ChatGroq(model=GROQ_MODEL, temperature=0)

    def retrieve(state: State):
        docs = store.similarity_search(state["question"], k=TOP_K)
        return {"context": "\n".join(d.page_content for d in docs)}

    def generate(state: State):
        reply = model.invoke([
            ("system", "Answer only using the supplied context. "
                       "If it lacks the answer, say you don't know."),
            ("human", f"Context: {state['context']}\nQuestion: {state['question']}"),
        ])
        return {"answer": reply.content}

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
