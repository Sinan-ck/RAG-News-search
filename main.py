import streamlit as st

# LangChain imports
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# =========================================================
# 1. STREAMLIT UI
# =========================================================

st.title("News Research Tool 📰")

st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")


# =========================================================
# 2. PROCESS URLs
# =========================================================

if process_url_clicked:

    # Remove empty URLs
    urls = [url for url in urls if url.strip()]

    if len(urls) == 0:

        st.error("Please enter at least one URL.")

    else:

        # -------------------------------------------------
        # Load webpages
        # -------------------------------------------------

        with st.spinner("Loading webpages..."):

            loader = UnstructuredURLLoader(
                urls=urls
            )

            documents = loader.load()

        st.success(
            f"Loaded {len(documents)} document(s)"
        )


        # -------------------------------------------------
        # Split documents
        # -------------------------------------------------

        with st.spinner("Splitting documents..."):

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=20
            )

            docs = splitter.split_documents(documents)

        st.success(
            f"Created {len(docs)} chunks"
        )


        # -------------------------------------------------
        # Create embeddings
        # -------------------------------------------------

        with st.spinner("Creating embeddings..."):

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )


        # -------------------------------------------------
        # Create FAISS vector store
        # -------------------------------------------------

        with st.spinner("Creating FAISS vector database..."):

            vectorstores = FAISS.from_documents(
                docs,
                embedding=embeddings
            )


        # Store vectorstore in Streamlit session
        st.session_state.vectorstores = vectorstores

        st.success("URLs processed successfully! ✅")


# =========================================================
# 3. QUESTION INPUT
# =========================================================

query = st.text_input(
    "Ask a question about the articles:"
)


# =========================================================
# 4. SEARCH + LLM
# =========================================================

if query:

    if "vectorstores" not in st.session_state:

        st.warning(
            "Please enter URLs and click 'Process URLs' first."
        )

    else:

        # -------------------------------------------------
        # Retrieve relevant chunks using FAISS
        # -------------------------------------------------

        results = st.session_state.vectorstores.similarity_search(
            query,
            k=3
        )


        # -------------------------------------------------
        # Create context
        # -------------------------------------------------

        context = "\n\n".join(
            result.page_content
            for result in results
        )


        # -------------------------------------------------
        # Create prompt
        # -------------------------------------------------

        prompt = ChatPromptTemplate.from_template("""
Answer the question using only the following context.

Context:
{context}

Question:
{question}

Answer:
""")


        # -------------------------------------------------
        # Create Ollama LLM
        # -------------------------------------------------

        llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0
        )


        # -------------------------------------------------
        # Send prompt to Ollama
        # -------------------------------------------------

        final_prompt = prompt.format(
            context=context,
            question=query
        )

        response = llm.invoke(final_prompt)


        # -------------------------------------------------
        # Display answer
        # -------------------------------------------------

        st.subheader("Answer")

        st.write(response.content)


        # -------------------------------------------------
        # Show retrieved chunks
        # -------------------------------------------------

        with st.expander("View retrieved chunks"):

            for i, result in enumerate(results):

                st.write(
                    f"### Chunk {i + 1}"
                )

                st.write(result.page_content)

                st.write("---")