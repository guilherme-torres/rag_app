import faiss
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama
from fastapi import FastAPI
from .config.ollama_config import OllamaConfig
from .config.neo4j_config import Neo4jConfig
from .rag import RagPipeline
from .utils.download_ollama_models import DownloadOllamaModels

app = FastAPI()

DownloadOllamaModels(OllamaConfig()).execute()

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

graph_store = Neo4jGraph(
    url=Neo4jConfig().NEO4J_URI,
    username=Neo4jConfig().NEO4J_USERNAME,
    password=Neo4jConfig().NEO4J_PASSWORD
)

llm = ChatOllama(
    model=OllamaConfig().LLM_MODEL,
    base_url=OllamaConfig().OLLAMA_HOST,
    temperature=OllamaConfig().MODEL_TEMPERATURE
)

rag_pipeline = RagPipeline(
    vector_store=vector_store,
    graph_store=graph_store,
    llm=llm
)

@app.get("/")
def generate(query: str):
    rag_pipeline.ingest()
    # documents = rag_pipeline.retrieve(query)
    # response = rag_pipeline.generate(query, documents)
    return {"response": ""}