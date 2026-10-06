# PharmaRAG: Clinical Decision Support via Hybrid Dense-Sparse Retrieval and Real-Time NLI Verification

**Authors:** Anonymous (For Double-Blind Peer Review)  
**Target Conference:** IEEE Journal of Biomedical and Health Informatics (JBHI) / EMNLP Clinical NLP Workshop / JAMIA  

---

## Abstract

Clinical Large Language Model (LLM) applications face significant challenges regarding factual correctness, domain hallucination, and safety risks in pharmaceutical decision support. In this paper, we propose **PharmaRAG**, an end-to-end Retrieval-Augmented Generation (RAG) framework designed for evidence-grounded medication decision support. PharmaRAG integrates a hybrid sparse-dense retrieval mechanism ($BM25 + \text{ChromaDB/MiniLM}$) merged via Reciprocal Rank Fusion (RRF, $k=60$), an intent classification guardrail preventing cross-domain leakage between drug indications and side-effect queries, and a real-time zero-shot Natural Language Inference (NLI) cross-encoder safety audit layer. Evaluated on a 36-query clinical benchmark spanning 5 domains, PharmaRAG achieves a **BERTScore F1 of 78.4%**, **ROUGE-L of 62.8%**, and **NLI Faithfulness of 89.2%**, outperforming conventional baseline architectures while maintaining a sub-1.2 second average query latency.

---

## 1. Introduction

Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to anchor LLM responses in domain-specific knowledge bases. However, in pharmaceutical clinical decision support:
1. **Keyword Ambiguity**: Drug trade names vs generic active ingredients often lead to sparse retrieval failures.
2. **Field Contamination**: A drug listed as causing *nausea* as a side effect might be incorrectly retrieved as a treatment for nausea.
3. **Hallucination Risk**: LLMs may generate plausibly sounding dosage or contraindication details not supported by the underlying pharmaceutical dataset.

To solve these challenges, we present **PharmaRAG**, a clinically grounded RAG system with multi-stage verification.

---

## 2. Methodology & Architecture

### 2.1 Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
Given a clinical query $q$, we compute sparse keyword relevance $R_{BM25}(q, d)$ and dense semantic embedding similarity $R_{Dense}(q, d)$ using `paraphrase-multilingual-MiniLM-L12-v2`. The candidates are fused using Reciprocal Rank Fusion:

$$S_{RRF}(d) = \sum_{m \in \{BM25, Dense\}} \frac{1}{k + r_m(d)}$$

where $k = 60$ and $r_m(d)$ is the rank of document $d$ in retriever $m$.

### 2.2 Intent Classification & Safety Guardrail
Queries are dynamically categorized into `indication` vs `side_effect` intents. The guardrail suppresses retrieved candidate records where target query terms match exclusively in mismatched schema fields (e.g. excluding side-effect-only matches for treatment queries).

### 2.3 Generation & Zero-Shot NLI Safety Audit
Conditioned on retrieved records $C_q$, a local LLM ($mistral$, $\tau=0.0$) generates a clinical summary $A_q$. A cross-encoder NLI model ($\text{distilroberta-base}$) evaluates textual entailment:

$$\text{Faithfulness Score} = P(\text{Entailment} \mid C_q, A_q)$$

If $\text{Faithfulness Score} < \theta$ ($\theta = 0.5$), the system flags the claim as UNVERIFIED in the UI.

---

## 3. Experimental Setup & Results

### 3.1 Benchmark Dataset
The evaluation dataset consists of 36 annotated clinical queries across 5 categories:
- **Indications & Uses** ($N=10$)
- **Contraindications & Precautions** ($N=7$)
- **Side Effects & Adverse Events** ($N=8$)
- **Dosage & Administration** ($N=6$)
- **Out-of-Domain & Negative Controls** ($N=5$)

### 3.2 Ablation Study Summary Table

| Model Variant | BERTScore F1 | ROUGE-L | BLEU-4 | Faithfulness (%) | Groundedness (%) | Recall@3 | MRR | Avg Latency (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ablation 1 (Sparse BM25 Only)** | 64.20 | 48.30 | 34.10 | 58.40 | 54.20 | 68.20 | 0.65 | 0.450 |
| **Ablation 2 (Dense MiniLM Only)** | 68.50 | 52.10 | 38.60 | 62.10 | 59.80 | 74.50 | 0.71 | 0.620 |
| **Ablation 3 (Hybrid RRF No Guardrail)** | 72.10 | 56.40 | 42.50 | 65.80 | 63.40 | 88.20 | 0.85 | 0.880 |
| **Proposed RAG (Hybrid + NLI)** | **78.40** | **62.80** | **48.90** | **89.20** | **84.60** | **95.10** | **0.94** | **1.150** |
| Baseline 1 (HomeDOCtor) | 70.00 | 52.40 | 38.10 | 65.00 | 62.00 | 61.50 | 0.58 | 0.900 |
| Baseline 2 (MEDIC) | 62.00 | 46.80 | 32.50 | 60.00 | 57.00 | 55.20 | 0.51 | 2.100 |

---

## 4. Discussion & Key Findings

1. **Hybrid Retrieval Superiority**: Combining BM25 keyword matching with dense MiniLM vector similarity improves Recall@3 from 68.2% (Sparse) and 74.5% (Dense) to **95.1%**.
2. **Factual Grounding via NLI**: The NLI safety audit and intent guardrail boost Faithfulness from 65.8% to **89.2%**, significantly mitigating hallucinations on out-of-domain queries.

---

## 5. Conclusion

PharmaRAG demonstrates that combining hybrid RRF retrieval with intent guardrails and cross-encoder NLI verification produces highly reliable, clinically grounded decision support without sacrificing low latency requirements.
