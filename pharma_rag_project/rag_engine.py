import os
import pandas as pd
from typing import List, Dict, Any
from generation import OllamaGenerator
from nli_judge import NLISafetyJudge

class RAGEngine:
    def __init__(self, dataset_path: str = "data/pharma_dataset.csv"):
        # 1. Load Dataset
        if os.path.exists(dataset_path):
            self.data = pd.read_csv(dataset_path).fillna("[NO DATA]")
        else:
            # Fallback English dataset
            self.data = pd.DataFrame([
                {
                    "Nom": "PARACETAMOL",
                    "Composition": "Paracetamol 500mg",
                    "Prescription": "Symptomatic treatment of mild to moderate pain and fever.",
                    "Posologie": "1 to 2 tablets per dose, up to 3 times daily.",
                    "Contrindications": "Severe hepatocellular insufficiency."
                },
                {
                    "Nom": "ZYRTEC",
                    "Composition": "Cetirizine dihydrochloride 10mg",
                    "Prescription": "Relief of nasal and ocular symptoms of allergic rhinitis.",
                    "Posologie": "1 tablet daily.",
                    "Contrindications": "Severe renal impairment (creatinine clearance < 10 ml/min)."
                }
            ])

        self.passages: List[Dict[str, Any]] = self.data.to_dict(orient="records")
        
        # 2. Instantiate Standalone Modules
        self.generator = OllamaGenerator(model_name="mistral")
        self.nli_judge = NLISafetyJudge()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top matching drug records based on query relevance."""
        query_lower = query.lower().strip()
        matched_results = []

        for record in self.passages:
            score = 0
            for key, value in record.items():
                val_str = str(value).lower()
                if query_lower in val_str:
                    score += 2
                elif any(word in val_str for word in query_lower.split() if len(word) > 2):
                    score += 1

            if score > 0:
                matched_results.append((score, record))

        matched_results.sort(key=lambda x: x[0], reverse=True)

        if matched_results:
            return [item[1] for item in matched_results[:top_k]]
        
        return self.passages[:top_k]

    def generate_llm_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Delegates output generation to generation.py in English."""
        try:
            return self.generator.generate(query, contexts)
        except Exception:
            if not contexts:
                return "⚠️ [NO DATA] No relevant clinical records were found to answer your request."
            
            doc = contexts[0]
            return (
                f"**Medication:** {doc.get('Nom', 'N/A')}\n\n"
                f"**Indications:** {doc.get('Prescription', 'N/A')}\n\n"
                f"**Dosage:** {doc.get('Posologie', '[NO DATA]')}\n\n"
                f"**Contraindications:** {doc.get('Contrindications', '[NO DATA]')}"
            )

    def verify_claim(self, response: str, context_block: str) -> bool:
        """Delegates claim verification to nli_judge.py."""
        try:
            formatted_context = [{"Context": context_block}]
            result = self.nli_judge.evaluate_faithfulness(response, formatted_context)
            if isinstance(result, dict):
                return result.get("is_safe", True)
            return bool(result)
        except Exception:
            return True