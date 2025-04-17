import streamlit as st
from utils import load_llm, set_prompt, FAISS_INDEX  # ← bring in what you need
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

def get_chain(simple: bool):
    # 1) load your embeddings & FAISS index
    embeddings = OllamaEmbeddings(model="initium/law_model")
    db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

    # 2) load the LLM and build the right prompt
    llm = load_llm()
    qa_prompt = set_prompt(simple)           # ← here’s where `simple` comes in

    # 3) assemble the RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )

def main():
    st.set_page_config(page_title="Guardian AI")
    st.title("Guardian AI")

    # —— style toggle goes here ——
    simple = st.checkbox("📖 Plain English")

    # —— and *only now* do we build the chain with that flag ——
    chain = get_chain(simple)

    # your fill‑in‑the‑blank hints
    st.markdown(
        "**💡 Try questions like:**  \n"
        ""
        "- _What does “__________” mean?_  \n"
        "- _Which section defines “__________”?_  \n"
        "- _Summarize “__________”._"
    )

    user_input = st.text_input(
        "You:",
        placeholder="e.g. What does “adoptee” mean?"
    )
    if user_input:
        # if they want simpler language, tack on that instruction
        query = user_input + ("  Please answer in plain, simple language." if simple else "")
        st.markdown(f"**You:** {user_input}  {'(simple)' if simple else ''}")
        bot_output = chain(query)["result"]
        st.markdown(f"**Guardian AI:** {bot_output}")

if __name__ == "__main__":
    main()
