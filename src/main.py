import os
import uuid
from typing import TypedDict
import faiss
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
import jwt
from jwt.exceptions import PyJWTError
from .config.ollama_config import OllamaConfig
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

llm = ChatOllama(
    model=OllamaConfig().LLM_MODEL,
    base_url=OllamaConfig().OLLAMA_HOST,
    temperature=OllamaConfig().MODEL_TEMPERATURE
)

chat_message_history = MongoDBChatMessageHistory(
    session_id='test',
    connection_string=f'mongodb://{os.getenv('MONGODB_USER')}:{os.getenv('MONGODB_PASSWORD')}@mongo:27017',
    database_name="chat_db",
    collection_name="chat_histories",
)

rag_pipeline = RagPipeline(
    vector_store=vector_store,
    llm=llm
)

class Attachment(TypedDict):
    content_type: str
    language: str
    content: str
    content_length: int

class RequestBody(BaseModel):
    assunto: str | None
    numero_processo: str
    data: str
    attachment: Attachment
    sistema: str
    numero_documento: str
    subassunto: str | None
    mimetype: str
    nucleo: str | None

def verify_token(request: Request) -> bool:
    authorization = request.headers['Authorization']
    token = ''.join(authorization.split('Bearer '))
    try:
        jwt.decode(token, os.getenv('JWT_SECRET'), algorithms=[os.getenv('JWT_ALGORITHM')])
        return True
    except PyJWTError as e:
        print(e)
        return False

@app.get("/")
def generate(body: RequestBody, query: str, authorized: bool = Depends(verify_token)):
    if not authorized:
        raise HTTPException(status_code=403, detail='Invalid or expired token')
    ids = rag_pipeline.ingest(body)
    documents = rag_pipeline.retrieve(query)
    response = rag_pipeline.generate(query, documents)
    rag_pipeline.clear_storage(ids=ids)
    chat_message_history.add_user_message(query)
    chat_message_history.add_ai_message(response)
    return {"response": response}