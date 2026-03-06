# RAG Pipeline with Automated Quality Testing

Production-ready Retrieval-Augmented Generation system with comprehensive test suite and reproducible quality metrics.

## 🎯 Project Highlights

- **Perfect Quality Metrics:** Faithfulness 1.000, Answer Quality 1.000, Context Precision 1.000, Context Recall 1.000
- **Systematic Optimization:** Debugged retrieval failures, optimized chunking (200→500 tokens), eliminated hallucinations (67%→0%)
- **Comprehensive Testing:** 27 automated tests (10 integration/unit + 17 adversarial)
- **Real-World Debugging:** Fixed 3 production failure modes through systematic root cause analysis

## 📊 Quality Scorecard

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Faithfulness** | 1.000 | >0.85 | ✅ Exceeds |
| **Answer Quality** | 1.000 | >0.90 | ✅ Exceeds |
| **Context Precision** | 1.000 | >0.75 | ✅ Exceeds |
| **Context Recall** | 1.000 | >0.75 | ✅ Exceeds |

*Note: Most production RAG systems score 0.75-0.85. Perfect 1.000 scores achieved through systematic optimization.*

## 🚀 Tech Stack

### Option 1: Local (Ollama)
- **LLM:** LLaMA 3.2 (via Ollama - local inference)
- **Embeddings:** Local embeddings model
- **Framework:** LangChain (RAG orchestration)
- **Vector DB:** ChromaDB (local development)

### Option 2: Cloud (HuggingFace + Groq)
- **LLM:** LLaMA 3.1 8B (via Groq API - cloud inference)
- **Embeddings:** HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Framework:** LangChain (RAG orchestration)
- **Vector DB:** ChromaDB (local development)

### General Stack
- **Testing:** pytest (27 tests with mocking and adversarial scenarios)
- **Evaluation:** RAGAS concepts (faithfulness, precision, recall, relevancy)

## 📺 Demo Video

[🎥 Watch 5-minute walkthrough](https://loom.com/your-link-here) *(Add link after recording)*

## 🧪 Test Coverage
```
tests/
├── test_rag_basic.py            (integration tests - real LLM calls)
├── test_rag_mocked_basic.py     (unit tests - mocked, fast)
└── test_adversarial.py          (adversarial tests - designed to break system)
```

**Test Categories:**
- ✅ **Integration Tests:** End-to-end RAG validation with real Ollama calls
- ✅ **Unit Tests (Mocked):** Logic validation without LLM calls (< 1 second)
- ✅ **Adversarial Tests:** Hallucination detection, scope violations, prompt injection attempts

## 🔍 Key Learnings

### 1. Retrieval Debugging (Day 2)
**Problem:** Question "What is the refund policy?" returned "I don't know" despite answer being in documents.

**Root Cause:** ChromaDB accumulated stale duplicate data from multiple script runs. Same irrelevant chunk appeared 3 times in top-3 results.

**Solution:** Implemented clean collection rebuilds before indexing. In production: use upsert-based indexing pipeline.

### 2. Chunking Optimization (Day 3)
Systematic experiment testing 200, 500, and 1000 token chunk sizes:

| Chunk Size | Success Rate | Issue |
|------------|-------------|-------|
| 200 tokens | 60% | Context fragmentation - policy split across chunks |
| **500 tokens** | **100%** | **Optimal - selected** |
| 1000 tokens | 100% | Verbose/contradictory answers from too much context |

### 3. Prompt Engineering (Day 4)
Measured hallucination rates across 3 prompt variants:

| Prompt Type | Hallucination Rate | Result |
|-------------|-------------------|---------|
| No guardrail | 67% | Made up answers to unknowable questions |
| Weak ("try to answer") | 33% | Inconsistent abstention |
| **Strong ("ONLY answer")** | **0%** | **Selected - clean abstention** |

### 4. Adversarial Testing (Day 8)
Built 17 test cases designed to break the system. **Found 1 vulnerability:** System was generating Python code when asked, violating scope. Fixed by adding explicit "do NOT write code" rule to system prompt.

## 🏃 Quick Start

### Prerequisites

**Option 1 (Local - Ollama):**
- Python 3.11+
- Ollama installed ([ollama.com](https://ollama.com))

**Option 2 (Cloud - Groq API):**
- Python 3.11+
- Groq API key from [console.groq.com](https://console.groq.com) (free tier available)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline

# Create virtual environment
python -m venv venv
# Activate virtualenv
# On macOS / Linux:
source venv/bin/activate
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# Or (cmd.exe):
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# For Option 1 (Ollama - local):
ollama pull llama3.2

# For Option 2 (Groq - cloud):
# Create .env file and add:
# GROQ_API_KEY=your_key_here
```

### Run Pipeline

**Local Implementation (Ollama):**
```bash
# Run RAG pipeline with Ollama
python rag_pipeline.py
```

**Cloud Implementation (Groq + HuggingFace):**
```bash
# Run RAG pipeline with Groq API
python rag_huggingFace.py
```

**Testing & Evaluation:**
```bash
# Run all tests
pytest tests/ -v

# Run only fast mocked tests
pytest tests/test_rag_mocked_basic.py -v

# View quality report
python quality_report.py
```

## 📂 Project Structure
```
genai-rag-pipeline/
├── rag_pipeline.py                # Main RAG implementation (local Ollama)
├── rag_huggingFace.py             # HuggingFace + Groq implementation (cloud API)
├── quality_report.py              # Quality metrics summary
├── golden_dataset.py              # Benchmark Q&A pairs
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (Groq API key)
├── sampledocs/                    # Knowledge base documents
│   ├── refund.txt
│   ├── leave.txt
│   └── onboarding.txt
└── tests/
    ├── conftest.py                # pytest fixtures
    ├── test_rag_basic.py          # Integration tests
    ├── test_rag_mocked_basic.py   # Unit tests (mocked, fast)
    └── test_adversarial.py        # Adversarial tests
```

## 📈 Before vs After Optimization

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Hallucination Rate | 67% | 0% | Eliminated through strong guardrails |
| Retrieval Failures | 40% | 0% | Fixed via chunking optimization |
| Answer Quality | 60% | 100% | Systematic debugging + testing |
| Test Suite Speed | 85s | <1s | Added mocked unit tests |

## 🔄 Implementation Comparison

| Aspect | Ollama (Local) | Groq (Cloud) |
|--------|--|--|
| **Inference** | Local GPU/CPU | Cloud API |
| **Setup** | Requires Ollama installation | Just API key needed |
| **Speed** | Depends on hardware | Optimized cloud infrastructure |
| **Privacy** | Data stays local | Data sent to Groq |
| **Cost** | Free (self-hosted) | Free tier + paid usage |
| **Embeddings** | Local model | HuggingFace (sentence-transformers) |

## 🎓 Skills Demonstrated

- **RAG Architecture:** End-to-end pipeline design (ingestion → chunking → embedding → retrieval → generation)
- **Quality Evaluation:** RAGAS-style metrics (faithfulness, precision, recall, relevancy)
- **Systematic Debugging:** Root cause analysis for retrieval failures, context fragmentation, hallucinations
- **Test Automation:** pytest with fixtures, mocking, parametrization, adversarial scenarios
- **Prompt Engineering:** Experimental approach to guardrail optimization with measurable outcomes
- **LangChain:** LCEL chains, retrievers, prompt templates, document loaders, text splitters
- **Multiple Implementations:** Local (Ollama) vs Cloud-based (Groq API) RAG pipelines
- **HuggingFace Integration:** Lightweight embeddings models for language understanding

## 🚧 Future Enhancements

- [ ] Add benchmarking between Ollama vs Groq implementations (latency, accuracy trade-offs)
- [ ] Migrate to Azure OpenAI + Azure AI Search (cloud deployment)
- [ ] Expand golden dataset from 5 to 50+ questions
- [ ] Add Flask API wrapper for production-like serving
- [ ] Implement CI/CD with GitHub Actions (automated testing on PR)
- [ ] Add Cosmos DB for conversation history (multi-turn chat)
- [ ] Deploy as Azure Function with Application Insights monitoring
- [ ] Add support for more LLM providers (OpenAI, Claude, etc.)

## 📧 Contact

Shubhangi Ajegaonkar
- LinkedIn: https://www.linkedin.com/in/shubhangi-ajegaonkar-62aa76aa/
- Email: shubhangi.ajegaonkar@gmail.com


---

*Built as part of GenAI testing specialization. Open to GenAI Tester / AI QA Engineer opportunities.*