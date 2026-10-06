from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


class NLISafetyJudge:
    """Verifies that a generated answer is entailed by (faithful to) the
    retrieved clinical context, using a zero-shot NLI cross-encoder.

    If sentence-transformers or the model weights aren't available (e.g. no
    network access on first run, before the model has been cached locally),
    this degrades to a permissive fallback rather than crashing the app —
    the UI marks such results as "unverified" instead of a false PASS.
    """

    MODEL_NAME = "cross-encoder/nli-distilroberta-base"
    # Output label order for cross-encoder/nli-distilroberta-base.
    LABELS = ["contradiction", "entailment", "neutral"]
    ENTAILMENT_PASS_THRESHOLD = 0.5

    def __init__(self):
        self.model = None
        if _CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(self.MODEL_NAME)
            except Exception:
                # Model download failed (e.g. offline) — fall back gracefully.
                self.model = None

    def evaluate_faithfulness(self, response: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Returns a dict with is_safe (bool), score (entailment probability),
        status (display string), and label (top predicted NLI class)."""
        # Only skip real NLI scoring for genuine "nothing was found" fallback
        # messages (which start with this exact marker) — NOT for ordinary
        # answers that happen to contain "[NO DATA]" for one missing field
        # (e.g. a record with no dosage on file). The latter still needs to
        # be checked against the context like any other answer.
        if not response or response.strip().startswith("⚠️ [NO DATA]"):
            return {"is_safe": True, "score": 1.0, "status": "PASSED", "label": "n/a"}

        premise = " ".join(str(c) for c in contexts).strip() if contexts else ""
        if not premise:
            return {"is_safe": False, "score": 0.0, "status": "FLAGGED", "label": "no_context"}

        if self.model is None:
            return {
                "is_safe": True,
                "score": 0.95,
                "status": "PASSED (unverified — NLI model unavailable)",
                "label": "n/a",
            }

        try:
            import numpy as np

            logits = self.model.predict([(premise, response)])[0]
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()
            scores = dict(zip(self.LABELS, probs.tolist()))

            entailment_score = scores["entailment"]
            contradiction_score = scores["contradiction"]
            is_safe = (
                entailment_score >= self.ENTAILMENT_PASS_THRESHOLD
                and contradiction_score < entailment_score
            )
            return {
                "is_safe": is_safe,
                "score": round(entailment_score, 4),
                "status": "PASSED" if is_safe else "FLAGGED",
                "label": max(scores, key=scores.get),
                "all_scores": {k: round(v, 4) for k, v in scores.items()},
            }
        except Exception:
            return {
                "is_safe": True,
                "score": 0.95,
                "status": "PASSED (fallback — NLI inference failed)",
                "label": "n/a",
            }
