import json
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

class RagPipeline:

    def __init__(self, vector_store: VectorStore, llm: BaseChatModel):
        self.vector_store = vector_store
        self.llm = llm
    

    def ingest(self) -> list[str]:
        data = None
        with open('src/document.json') as document_file:
            data = json.load(document_file)
        text_content = data['attachment']['content']
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250
        )
        chunks = text_splitter.create_documents([text_content])
        # print(chunks, len(chunks))
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        return self.vector_store.add_documents(documents=chunks, ids=uuids)


    def retrieve(self, query: str) -> list[Document]:
        return self.vector_store.similarity_search(query=query, k=5)


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
        # print(prompt)
        llm_response =  self.llm.invoke(prompt)
        return llm_response.content