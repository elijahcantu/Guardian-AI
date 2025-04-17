import os
import streamlit as st
from utils import load_llm, set_prompt, FAISS_INDEX
from ingest import embed_all
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


def get_chain(simple: bool):
    embeddings = OllamaEmbeddings(model="initium/law_model")
    db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()
    qa_prompt = set_prompt(simple)

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

    st.warning("⚠️ **This tool is for informational purposes only and cannot be used as legal advice.**")

    uploaded = st.file_uploader(
        "Upload one or more PDFs to add to the dataset",
        type="pdf",
        accept_multiple_files=True
    )
    if uploaded:
        os.makedirs("dataset", exist_ok=True)
        for pdf in uploaded:
            dest = os.path.join("dataset", pdf.name)
            with open(dest, "wb") as f:
                f.write(pdf.read())
        st.success(f"Saved {len(uploaded)} file(s) to dataset/, processing...")
        embed_new_files()

    simple = st.checkbox("📖 Plain English")

    chain = get_chain(simple)

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
        query = user_input + ("  Please answer in plain, simple language." if simple else "")
        st.markdown(f"**You:** {user_input}{' (simple)' if simple else ''}")

        bot_output = chain(query)["result"]

        if "legal advice" in user_input.lower():
            st.error("Guardian AI cannot provide legal advice. Please consult a qualified legal professional.")
        elif "information is not available" in bot_output.lower() or "further research" in bot_output.lower():
            st.info("The requested information is not available in the provided documents.")
        else:
            st.markdown(f"**Guardian AI:** {bot_output}")


if __name__ == "__main__":
    main()
