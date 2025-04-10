import faiss
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from fastapi import FastAPI
from .config.ollama_config import OllamaConfig
from .rag import RagPipeline

app = FastAPI()

embeddings = OllamaEmbeddings(
    model=OllamaConfig().EMBEDDING_MODEL,
    base_url=OllamaConfig().OLLAMA_HOST
)

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

llm = ChatOllama(
    model=OllamaConfig().LLM_MODEL,
    base_url=OllamaConfig().OLLAMA_HOST,
    temperature=OllamaConfig().MODEL_TEMPERATURE
)

rag_pipeline = RagPipeline(
    vector_store=vector_store,
    llm=llm
)

@app.get("/")
def generate(query: str):
    rag_pipeline.ingest()
    documents = rag_pipeline.retrieve(query)
    response = rag_pipeline.generate(query, documents)
    return {"response": response}