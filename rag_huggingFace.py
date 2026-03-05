"""
RAG Pipeline - Production-Ready Implementation

Author: Shubhangi Ajegaonkar
Date: February 2025
Purpose: GenAI Testing Portfolio Project
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Check if we have Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    # Use Groq (cloud API - no RAM issues)
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Fast, lightweight model
        temperature=0.7,
        api_key=groq_api_key
    )
    print("✅ Using Groq API (cloud)")
else:
    # Fallback: Use OpenAI-compatible API
    print("❌ No GROQ_API_KEY found in .env")
    print("Please add: GROQ_API_KEY=your_key_here")
    print("\\nGet free key from: https://console.groq.com")
    exit(1)

# ── Embeddings (Lightweight, runs locally) ───────────────────
print("Loading embeddings model (first time may take 1-2 min)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}  # Force CPU (no GPU needed)
)
print("✅ Embeddings model loaded")

# ── Load documents ────────────────────────────────────────────
def load_docs(folder="sampledocs"):
    """Load all text files from a directory into Document objects."""
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename)) as f:
                content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
    print(f"✅ Loaded {len(docs)} documents")
    return docs

# ── Chunk documents ───────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

raw_docs = load_docs()
chunks = splitter.split_documents(raw_docs)
print(f"✅ Split into {len(chunks)} chunks")

# ── Create vector store ───────────────────────────────────────
print("Creating vector store...")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="rag_fresh",
    persist_directory="./chroma_db"
)
print("✅ Vector store created")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── System prompt ─────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for answering questions about company policies.

RULES:
1. ONLY answer using the context below
2. If the answer is not in the context, say: "I don't know based on available information."
3. Never make up information
4. Do NOT write code, functions, or scripts
5. Do NOT answer questions unrelated to the provided policies

Context: {context}

Question: {question}
""")

# ── Format retrieved chunks ───────────────────────────────────
def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

# ── Build RAG chain ───────────────────────────────────────────
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Test questions ────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is the refund policy?",
        "How many days of annual leave do employees get?",
        "What happens in week 1 of onboarding?",
        "What is the CEO's name?",
        "Can I get a refund on a digital product?",
    ]

    print("\\n" + "="*50)
    print("RAG Pipeline - Testing")
    print("="*50)
    
    for q in questions:
        print(f"\\nQ: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer}")
        print("-"*40)
    
    print("\\n✅ All tests completed!")