import os
import streamlit as st
from utils import load_llm, set_prompt, FAISS_INDEX
from ingest import embed_all
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


def get_chain(simple: bool):
    # 1) load your embeddings & FAISS index
    embeddings = OllamaEmbeddings(model="initium/law_model")
    db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

    # 2) load the LLM and build the right prompt
    llm = load_llm()
    qa_prompt = set_prompt(simple)

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

    # —— NEW: PDF uploader ——
    uploaded = st.file_uploader(
        "📂 Upload one or more PDFs to add to the dataset",
        type="pdf",
        accept_multiple_files=True
    )
    if uploaded:
        os.makedirs("dataset", exist_ok=True)
        for pdf in uploaded:
            dest = os.path.join("dataset", pdf.name)
            with open(dest, "wb") as f:
                f.write(pdf.read())
        st.success(f"Saved {len(uploaded)} file(s) to dataset/, rebuilding index…")
        embed_all()

    # —— style toggle ——
    simple = st.checkbox("📖 Plain English")

    # —— build chain after knowing simple ——
    chain = get_chain(simple)

    # hints for fill-in-the-blank usage
    st.markdown(
        "**💡 Try questions like:**  \n"
        "- _What does “____” mean?_  \n"
        "- _Which section defines “____”?_  \n"
        "- _Summarize “____”._"
    )

    user_input = st.text_input(
        "You:",
        placeholder="e.g. What does “adoptee” mean?"
    )
    if user_input:
        # append simple-language instruction if requested
        query = user_input + ("  Please answer in plain, simple language." if simple else "")
        st.markdown(f"**You:** {user_input}{' (simple)' if simple else ''}")
        bot_output = chain(query)["result"]
        st.markdown(f"**Guardian AI:** {bot_output}")


if __name__ == "__main__":
    main()