# 📰 RAG News Search

An AI-powered News Research Tool built with **Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Ollama**.

The application allows users to provide news article URLs and ask questions about their content. The system retrieves relevant information using semantic search and generates answers using a local LLM.

## 🚀 Features

- 🌐 Load content from multiple webpage URLs
- ✂️ Split webpages into smaller chunks
- 🧠 Generate embeddings using Hugging Face
- 🔎 Semantic search using FAISS
- 🤖 Local answer generation using Ollama
- 💬 Ask questions about news articles
- 📚 View retrieved chunks
- 🔐 No OpenAI API key required

## 🏗️ Architecture

```text
News Article URLs
       ↓
UnstructuredURLLoader
       ↓
Web Content
       ↓
Text Chunking
       ↓
Hugging Face Embeddings
       ↓
FAISS Vector Store
       ↓
User Question
       ↓
Similarity Search
       ↓
Relevant Chunks
       ↓
Context + Question
       ↓
Ollama (Llama 3.2)
       ↓
Final Answer
🧠 RAG Pipeline
1. Load webpages

The user provides article URLs and the application extracts their content.

2. Split documents

Large webpages are divided into smaller chunks using RecursiveCharacterTextSplitter.

3. Create embeddings

Each chunk is converted into a vector using:

HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
4. Store vectors in FAISS

FAISS stores and indexes the embeddings for similarity search.

5. Retrieve relevant chunks

When the user asks a question:

results = vectorstores.similarity_search(
    query,
    k=3
)

FAISS retrieves the most relevant chunks.

6. Generate the answer

The retrieved chunks are passed as context to the local Ollama LLM.

🛠️ Technologies
Technology	Purpose
Python	Programming
Streamlit	Web UI
LangChain	RAG framework
Unstructured	Webpage extraction
Hugging Face	Embeddings
FAISS	Vector similarity search
Ollama	Local LLM
Llama 3.2	Answer generation
⚙️ Installation

Clone the repository:

git clone https://github.com/Sinan-ck/RAG-News-search.git
cd RAG-News-search

Create a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
🤖 Ollama Setup

Install Ollama and download the model:

ollama pull llama3.2:3b

Check the model:

ollama list

Make sure Ollama is running before starting the application.

▶️ Run the Application
streamlit run main.py

Then open the Streamlit URL shown in the terminal.

💡 Example

Enter news article URLs, click Process URLs, and ask questions such as:

What is the main topic of these articles?

The application retrieves relevant chunks from the articles and uses Ollama to generate the answer.

🔐 Local AI

The LLM runs locally through Ollama.

User
 ↓
Streamlit
 ↓
FAISS
 ↓
Relevant Context
 ↓
Ollama
 ↓
Llama 3.2
 ↓
Answer

No OpenAI API key is required.

📌 Future Improvements
Add source citations
Add conversation memory
Support document uploads
Improve webpage extraction
Add persistent FAISS storage
Improve UI design
Support multiple local LLMs
🎯 Learning Goals

This project helped me practice:

Retrieval-Augmented Generation (RAG)
Web scraping/data extraction
Document chunking
Text embeddings
Vector databases
Semantic search
FAISS
Prompt engineering
Ollama
LangChain
Streamlit
👨‍💻 Author

Muhammad Sinan

B.Tech Computer Science & Artificial Intelligence

GitHub: https://github.com/Sinan-ck
