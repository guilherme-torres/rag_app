import os
from datetime import datetime
import faiss
import requests
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

rag_pipeline = RagPipeline(
    vector_store=vector_store,
    llm=llm
)

class RequestBody(BaseModel):
    document_id: str
    chat_id: str
    query: str

def verify_token(request: Request) -> bool:
    authorization = request.headers['Authorization']
    token = ''.join(authorization.split('Bearer '))
    try:
        jwt.decode(token, os.getenv('JWT_SECRET'), algorithms=[os.getenv('JWT_ALGORITHM')])
        return token
    except PyJWTError as e:
        print(e)
        return False
    
class CustomMongoDBChatMessageHistory(MongoDBChatMessageHistory):
    def add_user_message(self, message: str) -> None:
        self.collection.insert_one({
            "session_id": self.session_id,
            "type": "human",
            "data": {"content": message},
            "timestamp": datetime.now().timestamp(),
        })

    def add_ai_message(self, message: str) -> None:
        self.collection.insert_one({
            "session_id": self.session_id,
            "type": "ai",
            "data": {"content": message},
            "timestamp": datetime.now().timestamp(),
        })

@app.post("/")
def generate(body: RequestBody, authorized: bool = Depends(verify_token)):
    if not authorized:
        raise HTTPException(status_code=403, detail='Invalid or expired token')
    
    chat_id = body.chat_id
    document_id = body.document_id
    search_document_endpoint = f'{os.getenv('GRACE_BACKEND_BASE_URL')}/api/v1/documents/search/{document_id}'
    headers = {
        'Authorization': f'Bearer {authorized}'
    }
    response = requests.get(search_document_endpoint, headers=headers)
    data = response.json()

    chat_message_history = CustomMongoDBChatMessageHistory(
        session_id=chat_id,
        connection_string=f'mongodb://{os.getenv('MONGODB_USER')}:{os.getenv('MONGODB_PASSWORD')}@mongo:27017',
        database_name="chat_db",
        collection_name="chat_histories",
    )

    ids = rag_pipeline.ingest(data['_source'])
    documents = rag_pipeline.retrieve(body.query)
    ai_response = rag_pipeline.generate(body.query, documents)
    rag_pipeline.clear_storage(ids=ids)
    chat_message_history.add_user_message(body.query)
    chat_message_history.add_ai_message(ai_response)

    return {"response": ai_response}