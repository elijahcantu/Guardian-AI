from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm  

DATASET = "dataset/"

FAISS_INDEX = "vectorstore/"

def embed_all():
    """
    Embed all files in the dataset directory with progress display
    """
    loader = DirectoryLoader(DATASET, glob="*.pdf", loader_cls=PyPDFLoader)
    
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = OllamaEmbeddings(model="initium/law_model")

    embedded_chunks = []
    for chunk in tqdm(chunks, desc="Embedding Chunks"):
        embedded_chunks.append(chunk)

    vector_store = FAISS.from_documents(embedded_chunks, embeddings)
    
    vector_store.save_local(FAISS_INDEX)
    print(f"Vector store saved at: {FAISS_INDEX}")

if __name__ == "__main__":
    embed_all()
