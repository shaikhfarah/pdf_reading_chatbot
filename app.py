import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List
import os
import requests
import time

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Streamlit config
st.set_page_config(page_title="PDF Chatbot", page_icon="📘", layout="wide")

# Custom LLM class for Groq API
class GroqLLM(LLM):
    model: str = "llama3-8b-8192"
    api_key: str = GROQ_API_KEY
    temperature: float = 0.3

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Groq API Error: {response.status_code} - {response.text}")

    @property
    def _llm_type(self) -> str:
        return "custom_groq_llm"

# 🧠 Extract all text from PDF pages
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:  # Removed page number display
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# ✂️ Split into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# 📦 Vector store with HuggingFace MiniLM
def create_vector_store(chunks):
    st.write(f"📊 Total Chunks to Embed: {len(chunks)}")
    start = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    st.success(f"✅ Vector store created in {round(time.time() - start, 2)} sec.")

# 🧠 QA chain using Groq LLM
def get_qa_chain():
    prompt_template = """
    Answer the question based only on the context below. If the answer isn't in the context, say so.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    groq_llm = GroqLLM()
    chain = load_qa_chain(llm=groq_llm, chain_type="stuff", prompt=prompt)
    return chain

# 🔍 Answer user's question
def answer_question(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    chain = get_qa_chain()
    result = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("📘 **Answer:**", result["output_text"])

# 🚀 Streamlit App Main
def main():
    with st.sidebar:
        st.title("📘 PDF Chatbot 🤖")
        st.markdown("Powered by **Groq API** + LLaMA3 🚀")
        st.markdown("👩‍💻 Built by **Farah Shaikh** ✨")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("🔍 Reading and indexing full PDF..."):
                    raw_text = get_pdf_text(pdf_docs)

                    if len(raw_text) > 200000:
                        st.warning("⚠️ This is a large document. Processing may take extra time.")

                    chunks = get_text_chunks(raw_text)
                    create_vector_store(chunks)
                st.success("✅ Done! You can now ask questions.")

    st.header("Ask a Question from the PDF 📄")
    user_question = st.text_input("Your Question")

    if user_question:
        try:
            answer_question(user_question)
        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
