from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

# Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# Embedding model
def watsonx_embedding():
    embed_params = {
        "temperature": 0.0,
        "max_tokens": 256,
    }
    model_id = "ibm/slate-125m-english-rtrvr"
    url = "https://us-south.ml.cloud.ibm.com"
    project_id = "skills-network"

    watsonx_embedding = WatsonxEmbeddings(
        model_id=model_id,
        url=url,
        project_id=project_id,
        params=embed_params,
    )
    return watsonx_embedding

# Vector database
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

# Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# LLM model
def get_llm():
    model_id = "mistralai/mixtral-8x7b-instruct-v01"
    parameters = {
        "temperature": 0.5,
        "max_tokens": 256,
    }
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm

# QA Chain
def retriever_qa(file, question):
    llm = get_llm()
    retr = retriever(file)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retr)
    answer = qa_chain.run(question)
    return answer