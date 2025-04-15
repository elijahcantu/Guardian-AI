# Importing Dependencies
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
DATASET = "dataset/"

FAISS_INDEX = "vectorstore/"

def embed_all():
    """
    Embed all files in the dataset directory
    """
    loader = DirectoryLoader(DATASET, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    chunks = documents
    embeddings = OllamaEmbeddings(model="initium/law_model")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX)

if __name__ == "__main__":
    embed_all()