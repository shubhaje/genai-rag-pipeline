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
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Check for Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file")
    print("Please add: GROQ_API_KEY=your_key_here")
    print("Get free key from: https://console.groq.com")
    exit(1)

# ══════════════════════════════════════════════════════════════
# INITIALIZE COMPONENTS
# ══════════════════════════════════════════════════════════════

print("🔧 Initializing RAG components...")

# LLM - Groq API (cloud-based, no RAM issues)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=groq_api_key
)
print("✅ LLM initialized (Groq)")

# Embeddings - HuggingFace (local, lightweight)
print("⏳ Loading embeddings model (first time may take 1-2 min)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
print("✅ Embeddings model loaded (HuggingFace)")

# ══════════════════════════════════════════════════════════════
# DOCUMENT LOADING & CHUNKING
# ══════════════════════════════════════════════════════════════

def load_docs(folder="sampledocs"):
    """Load all text files from a directory into Document objects."""
    docs = []
    if not os.path.exists(folder):
        print(f"❌ ERROR: Folder '{folder}' not found!")
        exit(1)
    
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
    
    if len(docs) == 0:
        print(f"❌ ERROR: No .txt files found in '{folder}'")
        exit(1)
    
    print(f"✅ Loaded {len(docs)} documents")
    return docs

# Text splitter - Optimal 500-token chunks with 50-token overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Load and chunk documents
raw_docs = load_docs()
chunks = splitter.split_documents(raw_docs)
print(f"✅ Split into {len(chunks)} chunks")

# ══════════════════════════════════════════════════════════════
# VECTOR STORE & RETRIEVER
# ══════════════════════════════════════════════════════════════

print("⏳ Creating vector store...")

# Create vector store with embeddings
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="rag_fresh",
    persist_directory="./chroma_db"
)
print("✅ Vector store created (ChromaDB)")

# Create retriever (top 3 most similar chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ══════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for answering questions about company policies.

RULES:
1. ONLY answer using the context below
2. If the answer is not in the context, say: "I don't know based on available information."
3. Never make up information
4. Do NOT write code, functions, or scripts
5. Do NOT answer questions unrelated to the provided policies
6. For questions with multiple parts, answer each part if information is available

Context: {context}

Question: {question}

Answer:
""")

# ══════════════════════════════════════════════════════════════
# FORMAT DOCS FUNCTION
# ══════════════════════════════════════════════════════════════

def format_docs(docs):
    """Combine retrieved documents into single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# ══════════════════════════════════════════════════════════════
# BUILD RAG CHAIN
# ══════════════════════════════════════════════════════════════

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG chain built successfully")
print("="*60)
print("RAG PIPELINE READY!")
print("="*60)

# ══════════════════════════════════════════════════════════════
# TEST QUESTIONS (only run when executing this file directly)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    questions = [
        "What is the refund policy?",
        "How many days of annual leave do employees get?",
        "What happens in week 1 of onboarding?",
        "What is the CEO's name?",
        "Can I get a refund on a digital product?",
    ]

    print("\n" + "="*60)
    print("TESTING RAG PIPELINE")
    print("="*60)
    
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Q: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer}")
        print("-"*60)
    
    print("\n✅ All tests completed!")