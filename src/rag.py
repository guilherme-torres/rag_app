import json
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer

class RagPipeline:

    def __init__(self, vector_store: VectorStore, graph_store, llm: BaseChatModel):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.llm = llm
    

    def ingest(self) -> list[str]:
        data = None
        with open('src/document.json') as document_file:
            data = json.load(document_file)
        content = data['attachment']['content']
        bs4_transformer = BeautifulSoupTransformer()
        text_content = bs4_transformer.extract_tags(
            html_content=content,
            tags=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'a', 'div', 'span', 'strong']
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250
        )
        chunks = text_splitter.create_documents([text_content])
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        print('gerando grafo de conhecimento ...')
        graph_documents = llm_transformer.convert_to_graph_documents(chunks)
        print(f"Nodes:{graph_documents[0].nodes}")
        print(f"Relationships:{graph_documents[0].relationships}")
        # uuids = [str(uuid4()) for _ in range(len(chunks))]
        # return self.vector_store.add_documents(documents=chunks, ids=uuids)


    def retrieve(self, query: str) -> list[Document]:
        return self.vector_store.similarity_search(query=query, k=10)


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
        llm_response =  self.llm.invoke(prompt)
        return llm_response.content