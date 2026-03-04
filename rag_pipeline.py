"""
RAG Pipeline - Production-Ready Implementation

Author: Shubhangi Ajegaonkar
Date: February 2025
Purpose: GenAI Testing Portfolio Project
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangSmith tracing (reads from .env automatically)
# Make sure .env has: LANGCHAIN_TRACING_V2=true
# Make sure .env has: LANGCHAIN_API_KEY=your_key

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── 1. LLM ──────────────────────────────────────────────────
llm = OllamaLLM(model="llama3.2:1b")

# ── 2. Embeddings ────────────────────────────────────────────
embeddings = OllamaEmbeddings(model="llama3.2:1b")

# ── 3. Load documents ────────────────────────────────────────
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
    print(f"Loaded {len(docs)} documents")
    return docs

# ── 4. Chunk documents ───────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

raw_docs = load_docs()
chunks = splitter.split_documents(raw_docs)
print(f"Split into {len(chunks)} chunks")

# ── 5. Create vector store ───────────────────────────────────
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="rag_fresh",
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── 6. System prompt ─────────────────────────────────────────
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

# ── 7. Format retrieved chunks ───────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── 8. Build RAG chain ───────────────────────────────────────
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 9. Test questions ────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is the refund policy?",
        "How many days of annual leave do employees get?",
        "What happens in week 1 of onboarding?",
        "What is the CEO's name?",
        "Can I get a refund on a digital product?",
    ]

    print("\n" + "="*50)
    print("RAG Pipeline - Testing")
    print("="*50)
    
    for q in questions:
        print(f"\nQ: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer}")
        print("-"*40)