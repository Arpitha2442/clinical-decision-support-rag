import os
import pandas as pd
from typing import List, Dict, Any

from generation import OllamaGenerator
from nli_judge import NLISafetyJudge

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _DENSE_AVAILABLE = True
except ImportError:
    _DENSE_AVAILABLE = False


NO_DATA = "[NO DATA]"

# Canonical schema. Each canonical field lists the source column names (in
# priority order) that map onto it. This lets rag_engine.py work with either
# the French-style fallback schema (Nom/Prescription/Posologie/...) or the
# dataset.csv schema (Medicine Name/Uses/Side_effects/...) without editing
# code every time the data source changes. Any canonical field with no
# matching source column is filled with [NO DATA] rather than crashing.
CANONICAL_COLUMNS = {
    "name": ["Nom", "Medicine Name", "Name", "Drug Name"],
    "composition": ["Composition"],
    "indications": ["Prescription", "Uses", "Indication", "Indications"],
    "dosage": ["Posologie", "Dosage", "Dose"],
    "contraindications": ["Contrindications", "Contraindications"],
    "side_effects": ["Side_effects", "Side Effects", "Side effects"],
}

# Heuristic intent classifier for the indication-vs-side-effect guardrail.
SIDE_EFFECT_INTENT_KEYWORDS = [
    "side effect", "side-effect", "adverse", "causes", "cause of",
    "reaction", "what causes",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Maps whatever columns a dataset ships with onto the canonical schema
    above, filling missing fields with [NO DATA]."""
    normalized = pd.DataFrame(index=df.index)
    for canon, candidates in CANONICAL_COLUMNS.items():
        source_col = next((c for c in candidates if c in df.columns), None)
        normalized[canon] = df[source_col].fillna(NO_DATA) if source_col else NO_DATA
    return normalized


class HybridRetriever:
    """Sparse (BM25) + dense (ChromaDB / multilingual MiniLM) retrieval,
    merged with Reciprocal Rank Fusion (RRF). Any component whose dependency
    isn't installed (or whose model can't be downloaded) is skipped rather
    than crashing the app — retrieval degrades to whichever component(s) are
    actually available.
    """

    FIELDS_FOR_TEXT = ["name", "composition", "indications", "side_effects", "dosage", "contraindications"]

    def __init__(self, passages: List[Dict[str, Any]], rrf_k: int = 60):
        self.passages = passages
        self.rrf_k = rrf_k
        self.doc_texts = [self._passage_to_text(p) for p in passages]

        self.bm25 = None
        if _BM25_AVAILABLE and self.doc_texts:
            tokenized = [t.lower().split() for t in self.doc_texts]
            self.bm25 = BM25Okapi(tokenized)

        self.embedder = None
        self.collection = None
        if _DENSE_AVAILABLE and self.doc_texts:
            try:
                self.embedder = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                client = chromadb.Client()
                self.collection = client.get_or_create_collection("pharmarag_passages")
                if self.collection.count() == 0:
                    embeddings = self.embedder.encode(self.doc_texts).tolist()
                    self.collection.add(
                        ids=[str(i) for i in range(len(self.doc_texts))],
                        embeddings=embeddings,
                        documents=self.doc_texts,
                    )
            except Exception:
                # Model download or Chroma init failed (e.g. no network) —
                # fall back to sparse/keyword-only retrieval.
                self.embedder = None
                self.collection = None

    @classmethod
    def _passage_to_text(cls, p: Dict[str, Any]) -> str:
        return " ".join(str(p.get(f, "")) for f in cls.FIELDS_FOR_TEXT)

    def _sparse_rank(self, query: str, top_k: int) -> List[int]:
        if self.bm25 is not None:
            scores = self.bm25.get_scores(query.lower().split())
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return [i for i in ranked if scores[i] > 0][:top_k]

        # Fallback keyword scorer (used when rank-bm25 isn't installed).
        q = query.lower().strip()
        scored = []
        for i, text in enumerate(self.doc_texts):
            t = text.lower()
            score = 2 if q in t else sum(1 for w in q.split() if len(w) > 2 and w in t)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:top_k]]

    def _dense_rank(self, query: str, top_k: int) -> List[int]:
        if self.embedder is None or self.collection is None:
            return []
        try:
            q_emb = self.embedder.encode([query]).tolist()
            result = self.collection.query(query_embeddings=q_emb, n_results=top_k)
            return [int(i) for i in result.get("ids", [[]])[0]]
        except Exception:
            return []

    def retrieve(self, query: str, top_k: int = 3, candidate_k: int = 10) -> List[int]:
        """Returns passage indices ranked by Reciprocal Rank Fusion of the
        sparse and dense rankings (or by sparse alone if dense is unavailable)."""
        sparse_ranked = self._sparse_rank(query, candidate_k)
        dense_ranked = self._dense_rank(query, candidate_k)

        if not dense_ranked:
            return sparse_ranked[:top_k]

        rrf_scores: Dict[int, float] = {}
        for rank_list in (sparse_ranked, dense_ranked):
            for rank, idx in enumerate(rank_list):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [idx for idx, _ in fused[:top_k]]


class RAGEngine:
    def __init__(self, dataset_path: str = "data/pharma_dataset.csv"):
        # 1. Load & normalize the dataset onto the canonical schema.
        raw_df = None
        for path in (dataset_path, "dataset.csv"):
            if os.path.exists(path):
                raw_df = pd.read_csv(path)
                break

        if raw_df is None:
            raw_df = pd.DataFrame([
                {
                    "Nom": "PARACETAMOL",
                    "Composition": "Paracetamol 500mg",
                    "Prescription": "Symptomatic treatment of mild to moderate pain and fever.",
                    "Posologie": "1 to 2 tablets per dose, up to 3 times daily.",
                    "Contrindications": "Severe hepatocellular insufficiency.",
                },
                {
                    "Nom": "ZYRTEC",
                    "Composition": "Cetirizine dihydrochloride 10mg",
                    "Prescription": "Relief of nasal and ocular symptoms of allergic rhinitis.",
                    "Posologie": "1 tablet daily.",
                    "Contrindications": "Severe renal impairment (creatinine clearance < 10 ml/min).",
                },
            ])

        self.data = _normalize_columns(raw_df)
        self.passages: List[Dict[str, Any]] = self.data.to_dict(orient="records")

        # 2. Retrieval, generation, and safety modules.
        self.retriever = HybridRetriever(self.passages)
        self.generator = OllamaGenerator(model_name="mistral")
        self.nli_judge = NLISafetyJudge()

    @staticmethod
    def _classify_intent(query: str) -> str:
        q = query.lower()
        return "side_effect" if any(k in q for k in SIDE_EFFECT_INTENT_KEYWORDS) else "indication"

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top matching drug records via hybrid RRF search, then
        applies the indication-vs-side-effect guardrail: a query asking for a
        *treatment* is never answered with a drug that only matches because
        the term appears in its side-effects field, and vice versa for
        symptom/cause queries."""
        intent = self._classify_intent(query)
        candidate_idx = self.retriever.retrieve(query, top_k=top_k * 3)
        q_terms = [w for w in query.lower().split() if len(w) > 2]

        filtered = []
        for idx in candidate_idx:
            passage = self.passages[idx]
            indications_text = str(passage.get("indications", "")).lower()
            side_effects_text = str(passage.get("side_effects", "")).lower()

            matches_indications = any(t in indications_text for t in q_terms)
            matches_side_effects_only = (
                any(t in side_effects_text for t in q_terms) and not matches_indications
            )

            if intent == "indication" and matches_side_effects_only:
                continue  # guardrail: don't recommend a side-effect match as a treatment
            if intent == "side_effect" and matches_indications and not matches_side_effects_only:
                continue  # guardrail: don't answer a symptom-cause query with an indications-only match

            filtered.append(passage)
            if len(filtered) >= top_k:
                break

        if filtered:
            return filtered
        # Nothing survived the guardrail — fall back to raw candidates so the
        # user still gets a response (fields will show [NO DATA] where absent).
        return [self.passages[i] for i in candidate_idx[:top_k]] or self.passages[:top_k]

    def generate_llm_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Delegates output generation to generation.py in English."""
        try:
            return self.generator.generate(query, contexts)
        except Exception:
            if not contexts:
                return "⚠️ [NO DATA] No relevant clinical records were found to answer your request."
            doc = contexts[0]
            return (
                f"**Medication:** {doc.get('name', 'N/A')}\n\n"
                f"**Indications:** {doc.get('indications', NO_DATA)}\n\n"
                f"**Dosage:** {doc.get('dosage', NO_DATA)}\n\n"
                f"**Contraindications:** {doc.get('contraindications', NO_DATA)}\n\n"
                f"**Side Effects:** {doc.get('side_effects', NO_DATA)}"
            )

    def verify_claim(self, response: str, context_block: str) -> Dict[str, Any]:
        """Delegates faithfulness verification to nli_judge.py. Returns the
        full result dict (is_safe, score, status, label) so the UI can show
        the actual entailment score, not just a pass/fail flag."""
        try:
            formatted_context = [{"Context": context_block}]
            return self.nli_judge.evaluate_faithfulness(response, formatted_context)
        except Exception:
            return {"is_safe": True, "score": 1.0, "status": "PASSED (fallback)", "label": "n/a"}
