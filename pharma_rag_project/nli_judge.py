from transformers import pipeline
from typing import Dict, List

class NLISafetyJudge:
    def __init__(self, model_name: str = "cross-encoder/nli-distilroberta-base"):
        """
        Loads zero-shot cross-encoder NLI model for verifying answer faithfulness.
        """
        try:
            self.nli_pipeline = pipeline(
                "text-classification", 
                model=model_name, 
                return_all_scores=True
            )
            self.is_active = True
        except Exception as e:
            print(f"Warning: Could not load local NLI model ({e}). Using rule-based fallback.")
            self.is_active = False

    def evaluate_faithfulness(self, generated_answer: str, context_documents: List[Dict]) -> Dict:
        """
        Verifies whether the generated LLM response is supported (Entailment) 
        or unsupported/contradicted by the retrieved context.
        """
        if not context_documents or generated_answer.startswith("⚠️"):
            return {
                "status": "UNVERIFIED",
                "label": "NO CONTEXT",
                "confidence": 0.0,
                "is_safe": False
            }

        # Build composite source context string
        source_text = " ".join([
            f"{doc.get('Nom', '')} {doc.get('Prescription', '')} {doc.get('Posologie', '')}" 
            for doc in context_documents
        ])

        if not self.is_active:
            # Heuristic fallback if NLI model fails to load
            has_keywords = any(word.lower() in source_text.lower() for word in generated_answer.split() if len(word) > 4)
            return {
                "status": "PASSED" if has_keywords else "FLAGGED",
                "label": "heuristic_entailment" if has_keywords else "heuristic_neutral",
                "confidence": 0.85 if has_keywords else 0.40,
                "is_safe": has_keywords
            }

        # Pair hypothesis (generated answer) with premise (retrieved source)
        # Sequence classification format for Hugging Face NLI: "premise </s></s> hypothesis"
        input_text = f"{source_text} </s></s> {generated_answer}"
        predictions = self.nli_pipeline(input_text)[0]

        # Standard NLI models output: LABEL_0 (Contradiction), LABEL_1 (Neutral), LABEL_2 (Entailment)
        # Parse top score label
        top_pred = max(predictions, key=lambda x: x['score'])
        label_mapping = {
            "LABEL_0": "Contradiction",
            "LABEL_1": "Neutral",
            "LABEL_2": "Entailment",
            "CONTRADICTION": "Contradiction",
            "NEUTRAL": "Neutral",
            "ENTAILMENT": "Entailment"
        }
        
        parsed_label = label_mapping.get(top_pred['label'].upper(), top_pred['label'])
        is_safe = parsed_label in ["Entailment", "LABEL_2"]

        return {
            "status": "PASSED" if is_safe else "FLAGGED",
            "label": parsed_label,
            "confidence": round(top_pred['score'], 4),
            "is_safe": is_safe
        }


if __name__ == "__main__":
    # Test execution
    judge = NLISafetyJudge()
    mock_context = [{
        "Nom": "ZYRTEC", 
        "Prescription": "Treatment of seasonal allergic rhinitis symptoms.", 
        "Posologie": "10mg once daily."
    }]
    
    # Grounded response test
    res1 = judge.evaluate_faithfulness("Zyrtec treat seasonal allergic rhinitis with 10mg daily.", mock_context)
    print("Test 1 Result (Valid Answer):", res1)

    # Hallucinated response test
    res2 = judge.evaluate_faithfulness("Zyrtec causes severe liver disease and should be taken 500mg per hour.", mock_context)
    print("Test 2 Result (Hallucination):", res2)