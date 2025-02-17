import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Configure Gemini AI with API Key
genai.configure(api_key=os.getenv("API_KEY"))

# AI model configuration
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=8192, timeout=None)

PERSIST_DIR = "chroma_store"  # Directory for storing the vector DB

# Extract texts from pdf file
def get_pdf_text(uploaded_file):
    if not uploaded_file:
        return None
    try:
        pdf_reader = PdfReader(uploaded_file)
        extracted_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return extracted_text if extracted_text.strip() else None
    except Exception as e:
        st.error(f"❌ Error extracting text: {e}")
        return None

# Splitting the large text into smaller chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

# Create the retriever and store it locally
def create_retriever(text_chunks):
    vectorstore = Chroma.from_texts(
        texts=text_chunks,
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=PERSIST_DIR  # This automatically saves the DB in local disk
    )
    st.success("✅ PDF processed and retriever stored locally!")


# Load the retriever from disk
def load_retriever():
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        st.error("Vector store not found. Please process a PDF first.")
        return None
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Answer the question based on user_input
def ask_question(user_input):
    # Fetch the retriever from local disk 
    retriever = load_retriever()
    if not retriever:
        return

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer, say so. "
        "Make sure to elaborate your answer for a better understanding.\n\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create a rag chain for answering questions
    # (i) First convert user_input into vector embeddings
    # (ii) Perform a similarity search algorithm to get the context
    # (iii) Inject the context with your question to AI to get a response
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": user_input})
    # print("💡 AI Answer:", response)
    print(response)
    print(response['answer'])
    answer = response['answer']
    st.write(answer)

def main():
    st.set_page_config(page_title="AI PDF Chat", layout="wide")
    st.title("💬 AI PDF Chat")
    st.markdown("Upload a PDF and interact with its contents using AI.")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.header("📂 Upload a PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if st.button("Submit"):
            if uploaded_file:
                extracted_text = get_pdf_text(uploaded_file)
                if extracted_text:
                    text_chunks = split_text_into_chunks(extracted_text)
                    create_retriever(text_chunks)  # Store retriever locally
                else:
                    st.warning("⚠️ No text extracted from the PDF.")
            else:
                st.warning("⚠️ Please upload a PDF file before clicking Submit.")

    # Chat section for asking questions
    st.subheader("🗨️ Chat with Your PDF")
    user_query = st.text_input("Ask a question about the PDF...", placeholder="Type your question here")

    if st.button("Get Answer"):
        if user_query:
            ask_question(user_query)  # Invoke chain and print answer to console
            
        else:
            st.warning("⚠️ Please enter a question.")


if __name__ == "__main__":
    main()
