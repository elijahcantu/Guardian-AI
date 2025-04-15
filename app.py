import streamlit as st
from utils import qa_pipeline

chain = qa_pipeline()


st.set_page_config(
    page_title="Guardian AI",  

)

def main():
    st.session_state.setdefault("chat_log", [])

    st.title('Guardian AI')

    user_input = st.text_input("You:")

    if user_input:
        st.markdown(f'**You:** {user_input}')
        bot_output = chain(user_input)['result']
        st.markdown(f'**Guardian AI:** {bot_output}')


        

if __name__ == "__main__":
    main()
