# 📘 PDF Chatbot

A powerful chatbot application that allows users to **ask questions from uploaded PDF documents** using the **Groq API** with **LLaMA 3**, **HuggingFace Embeddings**, and **FAISS Vector Store** — all wrapped in a user-friendly **Streamlit** interface.

---

## 🚀 Features

- 🧠 Ask questions from multiple PDF documents.
- ⚡ Powered by [Groq API](https://console.groq.com/) using LLaMA 3 (fast inference).
- 📄 Extracts and indexes all pages from uploaded PDFs.
- 🧩 Text chunking using LangChain’s `RecursiveCharacterTextSplitter`.
- 🔍 Similarity search via FAISS vector store.
- 🌐 Fully functional Streamlit web app.

---

![image](https://github.com/user-attachments/assets/fafbfa45-4314-43cc-a31e-2af7a692fda6)


## 🧰 Tech Stack

- `Python`
- `Streamlit`
- `Groq API` with `LLaMA3`
- `HuggingFace Embeddings`
- `FAISS` (for vector similarity)
- `LangChain`
- `PyPDF2`
- `.env` (to manage API keys securely)

---

## 📁 Project Structure
pdf-chatbot
┣ 📄 app.py
┣ 📄 requirements.txt
┣ 📄 .env
Inside .env, add:

GROQ_API_KEY=your_groq_api_key_here

🙋‍♀️ Author
Farah Shaikh
🎓 Final Year Engineering Student (AI & Data Science)
🔗 LinkedIn • 📧 shaikfaraha288@gmail.com

