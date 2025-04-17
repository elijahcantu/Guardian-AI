import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

FAISS_INDEX = "vectorstore/"

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def set_custom_prompt_template():
    system = SystemMessagePromptTemplate.from_template(
        """You are a legal assistant specializing in Michigan foster care and statutory law.
            Provide clear, fact-based answers strictly using the provided documents.
            Always cite relevant statutes when applicable.
            If the context is insufficient, , state explicitly that the information is not available in the provided documents.
            Do not make assumptions or provide legal advice under any circumstances."""
    )
    human = HumanMessagePromptTemplate.from_template(
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer strictly using only the context above.  \n"
    "- If you quote a statutory definition, include the exact section number.  \n"
    "- If the user is asking for a summary, summarize *only* what appears in the retrieved text.  \n"
    "- If the question requests legal advice, respond: “I’m sorry, but I can’t provide legal advice.”  \n"
    "- If the information is not present in the context, reply: “Information not found in the provided documents.”"
    )
    return ChatPromptTemplate.from_messages([system, human])



if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

def set_prompt(simple: bool):
    sys = (
        "You are a legal assistant specializing in Michigan foster care and statutory law. "
        "Provide clear, fact-based answers strictly using the provided documents. "
        "Always cite relevant statutes when applicable. "
        "If the context is insufficient, state explicitly that the information is not available in the provided documents. "
        "Do not make assumptions or provide legal advice under any circumstances. "
        + ("Always explain in plain, simple English." if simple else "")
    )
    human = HumanMessagePromptTemplate.from_template(
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer strictly using only the context above.  \n"
    "- If you quote a statutory definition, include the exact section number.  \n"
    "- If the user is asking for a summary, summarize *only* what appears in the retrieved text.  \n"
    "- If the question requests legal advice, respond: “I’m sorry, but I can’t provide legal advice.”  \n"
    "- If the information is not present in the context, reply: “Information not found in the provided documents.”"
    )
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(sys),
        human
    ])



def load_llm():
    """
    Load the Ollama model
    """
    return ChatOllama(model="initium/law_model")

def retrieval_qa_chain(llm, prompt, db):
    """
    Create the Retrieval QA chain
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_pipeline():
    """
    Create the QA pipeline with Ollama
    """
    embeddings = OllamaEmbeddings(model="initium/law_model") 

    db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()
    qa_prompt = set_custom_prompt_template()

    qa_chain = retrieval_qa_chain(llm, qa_prompt, db)

    return qa_chain

def main():
    if 'chat_log' not in st.session_state:
        st.session_state.chat_log = []

    user_input = st.text_
