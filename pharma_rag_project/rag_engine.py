import pandas as pd
import chromadb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGEngine:
    def __init__(self, csv_path="dataset.csv"):
        # 1. Load Multilingual Embedding Model
        print("Loading SentenceTransformer embedding model...")
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # 2. Load NLI Model for Post-Generation Safety Judge
        print("Loading NLI Safety Judge...")
        self.nli_judge = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
        
        # 3. Read & Preprocess CSV Dataset
        print("Loading CSV dataset...")
        self.df = pd.read_csv(csv_path).head(1000).fillna("[NO DATA]")
        self.passages = self._preprocess_passages(self.df)
        
        # 4. Initialize BM25 Sparse Keyword Search
        print("Indexing corpus into BM25 Keyword Search...")
        tokenized_corpus = [doc.lower().split() for doc in self.passages]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 5. Initialize ChromaDB In-Memory Vector Store
        print(f"Indexing {len(self.passages)} medication passages into ChromaDB...")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="medications")
        
        # Generate Embeddings & Add to Vector Store
        embeddings = self.embedder.encode(self.passages).tolist()
        ids = [str(i) for i in range(len(self.passages))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=self.passages,
            ids=ids
        )
        print("Vector store indexing complete!")

    def _preprocess_passages(self, df):
        """Standardizes CSV metadata fields into structured passage strings with explicit tokens."""
        passages = []
        for _, row in df.iterrows():
            med_name = str(row.get('Medicine Name', row.get('Name', row.get('name', 'Unknown'))))
            composition = str(row.get('Composition', row.get('Category', row.get('generic_name', '[NO DATA]'))))
            uses = str(row.get('Uses', row.get('Indication', row.get('use0', '[NO DATA]'))))
            side_effects = str(row.get('Side_effects', row.get('Dosage Form', '[NO DATA]')))
            
            text = (
                f"Medication: {med_name} | Composition/Category: {composition} | "
                f"Uses/Indications: {uses} | Side Effects/Form: {side_effects}"
            )
            passages.append(text)
        return passages

    def retrieve_dense(self, query: str, top_k: int = 5):
        """Dense Vector Similarity Search via ChromaDB."""
        query_vector = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k
        )
        return results['documents'][0]

    def retrieve_sparse(self, query: str, top_k: int = 5):
        """Sparse Keyword Search via BM25."""
        tokenized_query = query.lower().split()
        return self.bm25.get_top_n(tokenized_query, self.passages, n=top_k)

    def retrieve(self, query: str, top_k: int = 3):
        """
        Hybrid Retrieval: Merges Vector Search and BM25 Keyword Search
        using Reciprocal Rank Fusion (RRF).
        """
        dense_results = self.retrieve_dense(query, top_k=top_k * 2)
        sparse_results = self.retrieve_sparse(query, top_k=top_k * 2)
        
        rrf_scores = {}
        for rank, doc in enumerate(dense_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + (1.0 / (60 + rank + 1))
            
        for rank, doc in enumerate(sparse_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + (1.0 / (60 + rank + 1))
            
        sorted_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        return sorted_docs[:top_k]

    def generate_llm_response(self, query: str, retrieved_contexts: list) -> str:
        """
        Generates context-restricted responses using Ollama (Mistral/Llama3).
        Falls back to strict intent-aware parsing if Ollama is offline.
        """
        context_str = "\n---\n".join(retrieved_contexts)
        
        prompt = f"""You are a clinical decision support AI assistant.
Answer the user query using ONLY the provided medication context. Do not invent facts or extrapolate beyond this evidence.

CRITICAL SAFETY INSTRUCTIONS:
1. Pay strict attention to the distinction between "Uses/Indications" and "Side Effects".
2. If a condition (e.g., vertigo, headache, nausea) is listed ONLY under "Side Effects", do NOT recommend that drug as a treatment!
3. If no retrieved drug in the evidence explicitly treats the user's condition under "Uses/Indications", state clearly:
   "⚠️ No matching treatment found in the clinical records."

CONTEXT EVIDENCE:
{context_str}

USER QUERY: {query}

ANSWER:"""

        # Try local Ollama instance first
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0}
                },
                timeout=10
            )
            if response.status_code == 200:
                answer = response.json().get("response", "").strip()
                if answer:
                    return answer
        except Exception:
            pass  # Fallback to local template parser

        return self._fallback_intent_formatter(query, retrieved_contexts)

    def _fallback_intent_formatter(self, query: str, retrieved_contexts: list) -> str:
        """
        Strict fallback parser that verifies query keywords appear under 
        'Uses/Indications' before recommending any medication.
        """
        stop_words = {"what", "is", "the", "medicine", "used", "for", "drug", "take", "best", "treatment", "with", "can"}
        query_words = [w.lower().strip("?,.") for w in query.split() if w.lower().strip("?,.") not in stop_words and len(w) > 2]

        target_doc = None

        # Check strictly inside 'Uses/Indications:' section
        for doc in retrieved_contexts:
            if "Uses/Indications:" in doc:
                try:
                    uses_section = doc.split("Uses/Indications:")[1].split("|")[0].lower()
                    if any(word in uses_section for word in query_words):
                        target_doc = doc
                        break
                except IndexError:
                    continue

        # CRITICAL SAFETY GUARDRAIL RETURN
        if target_doc is None:
            terms_searched = ", ".join([f"'{w}'" for w in query_words]) if query_words else f"'{query}'"
            return (
                f"⚠️ **No matching treatment found in the clinical records.**\n\n"
                f"None of the retrieved medications are explicitly indicated to treat {terms_searched}. "
                f"Please consult a certified healthcare professional."
            )

        # Only executes if an explicit match was found in Uses/Indications
        doc_parts = target_doc.split(" | ")
        med_name = doc_parts[0].replace("Medication: ", "")
        comp = doc_parts[1].replace("Composition/Category: ", "")
        uses = doc_parts[2].replace("Uses/Indications: ", "")
        side_fx = doc_parts[3].replace("Side Effects/Form: ", "")

        return (
            f"Based on pharmaceutical database records, **{med_name}** ({comp}) "
            f"is indicated for **{uses}**. "
            f"Reported adverse reactions and side effects include: {side_fx}."
        )

    def verify_claim(self, generated_answer: str, context: str) -> bool:
        """NLI Entailment Verification to check claim alignment against retrieved context."""
        hypothesis = f"The statement '{generated_answer}' is directly supported by the retrieved context: {context}"
        result = self.nli_judge(hypothesis, candidate_labels=["supported", "unsupported"])
        return result['labels'][0] == "supported"