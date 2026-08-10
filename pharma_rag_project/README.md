# 💊 Clinical Decision Support RAG System

A hybrid Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **ChromaDB**, **BM25**, and **SentenceTransformers** that provides medical information retrieval with strict intent-aware safety guardrails and NLI-based factual verification.

---

## 🎯 Key Features

- **Hybrid Search Architecture (RRF)**: Combines **BM25 (sparse keyword search)** for exact drug brand names and active ingredient tokens with **ChromaDB + MiniLM (dense vector search)** for semantic symptom matching using **Reciprocal Rank Fusion**.
- **Indication vs. Side-Effect Guardrail**: Strictly separates `Uses/Indications` from `Side Effects`. If a query asks for a treatment (e.g., *"vertigo"*), the system refuses to recommend drugs where the symptom only appears as a side effect (e.g., *"dizziness"*).
- **Post-Generation NLI Judge**: Uses a zero-shot cross-encoder NLI model (`cross-encoder/nli-distilroberta-base`) to verify if the generated response is factually supported by the retrieved database context.
- **Local LLM & Fallback Engine**: Integrates with a local **Ollama** instance (Mistral/Llama 3) with an automated deterministic fallback parser if offline.

---

## 🏗️ Architecture Pipeline
[ User Query ]
│
├───> Sparse Search (BM25 Token Match) ──────────┐
│                                                ├──> [ Reciprocal Rank Fusion (RRF) ] ──> Top-3 Contexts
└───> Dense Search (ChromaDB + MiniLM Embeddings)┘                                             │
▼
[ Indication Guardrail Filter ]
│
▼
[ Verified Output + NLI Badge ] <────────── [ Zero-Shot NLI Entailment Judge ] <────────── [ LLM / Fallback Engine ]

## 🛠️ Tech Stack

- **Frontend / UI**: Streamlit
- **Vector Database**: ChromaDB
- **Keyword Search**: BM25 (`rank-bm25`)
- **Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **NLI Fact Checker**: `cross-encoder/nli-distilroberta-base`
- **Data Processing**: Pandas, Python 3.11+

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository and create a Python virtual environment:

```bash
git clone [https://github.com/your-username/pharma-rag-project.git](https://github.com/your-username/pharma-rag-project.git)
cd pharma-rag-project

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

2. Install Dependencies
Bash
.\venv\Scripts\python.exe -m pip install -r requirements.txt

3. Add DatasetEnsure your CSV dataset (e.g., dataset.csv) is placed in the root directory. Expected columns include:Medicine Name / NameComposition / CategoryUses / IndicationSide_effects4. Run the ApplicationLaunch the Streamlit web interface:Bash.\venv\Scripts\python.exe -m streamlit run app.py --server.fileWatcherType folder
🧪 Sample Evaluation Test CasesQueryExpected IndicationSafety Outcome"uses of paracetamol"Pain relief / Fever✅ Matched (Exact indication match)"medication that causes nausea"Side Effects field✅ Matched (Mapped accurately to adverse effects)"treatment for vertigo"Not in Indication⚠️ Safely Rejected (Prevented recommending statins listing dizziness as a side effect)
📄 DisclaimerThis application is built strictly for decision support research and educational demonstration using public pharmaceutical database records. It is not intended as a substitute for professional medical advice, diagnosis, or treatment.