import os
import faiss
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.config.ollama_config import OllamaConfig
from src.rag import RagPipeline
from src.utils.download_ollama_models import DownloadOllamaModels

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

print(f'Using {OllamaConfig().LLM_MODEL} model')
print(f'Using {OllamaConfig().EMBEDDING_MODEL} embedding model')

llm = ChatOllama(
    model=OllamaConfig().LLM_MODEL,
    base_url=OllamaConfig().OLLAMA_HOST,
    temperature=OllamaConfig().MODEL_TEMPERATURE
)

rag_pipeline = RagPipeline(
    vector_store=vector_store,
    llm=llm
)

fake_document_storage = {
    1: {
        "id": 1,
        "file_path": "/app/src/documents/ada_lovelace.txt"
    },
    2: {
        "id": 2,
        "file_path": "/app/src/documents/batalha_jenipapo.txt"
    },
    3: {
        "id": 3,
        "file_path": "/app/src/documents/maquina_turing.txt"
    },
    4: {
        "id": 4,
        "file_path": "/app/src/documents/teresina.txt"
    }
}

@app.get("/")
def generate(document_id: int, query: str):
    file_path = fake_document_storage.get(document_id, {}).get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="document not found")
    document_content = ""
    with open(file_path, 'r') as document_file:
        document_content = document_file.read()
    ai_response = rag_pipeline.generate(query, document_content)
    return {"response": ai_response}