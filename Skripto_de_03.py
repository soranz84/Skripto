import os
import streamlit as st
from streamlit_chat import message
from rag_de_02 import ChatPDF
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

st.set_page_config(page_title="ChatPDF")

PDF_FOLDER = os.path.dirname(__file__)  # Get script's folder
PDF_FILE = next((f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")), None)  # Find a PDF


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def ingest_pdf():
    """ Load and process the PDF automatically if embeddings do not exist. """
    persist_directory = "./chroma"  # Directory where embeddings are stored

    if os.path.exists(persist_directory) and os.listdir(persist_directory):  # Ensure folder is not empty
        try:
            # Load existing vector store
            st.session_state["assistant"].vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=FastEmbedEmbeddings()  # Ensure embedding function is passed
            )
            st.session_state["assistant"].retriever = st.session_state["assistant"].vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.5}
            )
            st.write("✅ Loaded existing embeddings from storage.")
        except Exception as e:
            st.error(f"⚠️ Error loading ChromaDB: {e}")
    else:
        # If no saved embeddings, ingest the PDF
        if PDF_FILE:
            pdf_path = os.path.join(PDF_FOLDER, PDF_FILE)
            st.session_state["assistant"].ingest(pdf_path)
            st.write("✅ PDF processed and embeddings saved.")

def page():
    if "assistant" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()
        ingest_pdf()  # Auto-load the PDF

    st.header(f"ChatPDF - {PDF_FILE if PDF_FILE else 'No PDF Found'}")

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
