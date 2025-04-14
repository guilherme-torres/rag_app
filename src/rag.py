import json
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import BeautifulSoupTransformer

class RagPipeline:

    def __init__(self, vector_store: VectorStore, llm: BaseChatModel):
        self.vector_store = vector_store
        self.llm = llm
    

    def ingest(self, data) -> list[str]:
        content = data.attachment['content']
        bs4_transformer = BeautifulSoupTransformer()
        text_content = bs4_transformer.extract_tags(
            html_content=content,
            tags=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'a', 'div', 'span', 'strong']
        )
        documents = [Document(page_content=text_content)]
        uuids = [str(uuid4()) for _ in range(len(documents))]
        return self.vector_store.add_documents(documents=documents, ids=uuids)

    
    def clear_storage(self, ids: list[str]) -> None:
        self.vector_store.delete(ids=ids)


    def retrieve(self, query: str) -> list[Document]:
        return self.vector_store.similarity_search(query=query, k=1)


    def generate(self, query: str, documents: list[Document]):
        prompt_template = PromptTemplate.from_template('''
        ## Instrução ##
        Você é um assistente jurídico especializado em responder perguntas com base exclusivamente nos documentos fornecidos.
        - Se a informação não estiver presente, responda apenas: **"Desculpe, esta informação não consta no documento"**
        - Não tente inferir ou supor respostas com base em conhecimento externo ou senso comum.
        ## Documentos ##
        {documents}
        ## Pergunta ##
        {query}
        Resposta:
        ''')
        prompt = prompt_template.invoke({
            "documents": '\n'.join([f' - {document.page_content}' for document in documents]),
            "query": query
        })
        llm_response = self.llm.invoke(prompt)
        return llm_response.content