import os
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

class ChatPDF:
    persist_directory = "./chroma"  # Folder where embeddings are stored

    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] Sie sind ein Assistent für die Beantwortung von Fragen. Nutzen Sie die folgenden Kontextinformationen, um die Frage zu beantworten.
Wenn Sie die Antwort nicht kennen, sagen Sie einfach, dass Sie es nicht wissen. Verwenden Sie maximal drei Sätze und fassen Sie sich kurz. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

        # 🔹 Try loading existing embeddings first
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.load_existing_embeddings()
        else:
            self.vector_store = None
            self.retriever = None
            self.chain = None

    def load_existing_embeddings(self):
        """ Load the saved vector store from disk """
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=FastEmbedEmbeddings()
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5}
        )
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ingest(self, pdf_file_path: str):
        """ Process and save embeddings from a new PDF """
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory=self.persist_directory  # Ensure persistence
        )
    
        # Save embeddings
        self.vector_store.persist()

        # Create the retriever and chain
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.5}
        )
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        """ Answer questions using stored embeddings """
        if not self.chain:
            return "⚠️ No document found! Please add a PDF first."
        return self.chain.invoke(query)

    def clear(self):
        """ Clear embeddings and reset state """
        self.vector_store = None
        self.retriever = None
        self.chain = None
