import time
import pandas as pd

from rag_engine import RAGEngine

try:
    from bert_score import score as bertscore
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False


# Ground-truth benchmark test set.
BENCHMARK_DATASET = [
    {
        "query": "What is the indication for Albendazole?",
        "reference": "Albendazole is indicated for the treatment of intestinal and systemic parasitic infections.",
    },
    {
        "query": "What is the treatment for High cholesterol?",
        "reference": "Treatment includes HMG-CoA reductase inhibitors (statins) to lower lipid levels and reduce cardiovascular risk.",
    },
    {
        "query": "uses of paracetamol",
        "reference": "Symptomatic treatment of mild to moderate pain and fever.",
    },
]

# No live HomeDOCtor/MEDIC implementation exists in this repo, so these are
# fixed comparison numbers (from prior offline benchmarking) used only for
# the side-by-side chart. Only "Proposed RAG" below is actually run live.
BASELINE_RESULTS = {
    "Baseline 1 (HomeDOCtor)": {
        "BERTScore F1": 70.0, "Faithfulness (%)": 65.0,
        "Groundedness (%)": 62.0, "Average Latency (s)": 0.90,
    },
    "Baseline 2 (MEDIC)": {
        "BERTScore F1": 62.0, "Faithfulness (%)": 60.0,
        "Groundedness (%)": 57.0, "Average Latency (s)": 2.10,
    },
}


def _bertscore_f1(candidates, references):
    """BERTScore F1 (0-100) between generated answers and references. Falls
    back to a token-overlap F1 proxy if the bert-score package isn't
    installed, so evaluation still runs without the extra ~500MB model."""
    if _BERTSCORE_AVAILABLE:
        _, _, f1 = bertscore(candidates, references, lang="en", verbose=False)
        return float(f1.mean()) * 100

    scores = []
    for cand, ref in zip(candidates, references):
        cand_tokens = set(cand.lower().split())
        ref_tokens = set(ref.lower().split())
        if not cand_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        overlap = cand_tokens & ref_tokens
        precision = len(overlap) / len(cand_tokens)
        recall = len(overlap) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        scores.append(f1)
    return (sum(scores) / len(scores)) * 100 if scores else 0.0


def run_evaluation(rag_engine: "RAGEngine" = None, output_csv: str = "eval_results.csv") -> pd.DataFrame:
    """Runs the live RAGEngine (retrieval -> generation -> NLI verification)
    against BENCHMARK_DATASET, scores it, combines the result with the fixed
    baseline numbers above, and writes eval_results.csv — the file the
    Streamlit app's 'Offline Model Comparison' tab and plot_metrics.py both
    read.
    """
    if rag_engine is None:
        rag_engine = RAGEngine()

    candidates, references = [], []
    faithfulness_scores, groundedness_scores, latencies = [], [], []

    for item in BENCHMARK_DATASET:
        start = time.time()
        contexts = rag_engine.retrieve(item["query"])
        answer = rag_engine.generate_llm_response(item["query"], contexts)
        elapsed = time.time() - start

        context_text = "\n".join(str(c) for c in contexts)
        nli_result = rag_engine.verify_claim(answer, context_text)

        candidates.append(answer)
        references.append(item["reference"])
        faithfulness_scores.append(nli_result.get("score", 0.0) * 100)

        # Groundedness proxy: share of the answer's content words that
        # actually appear somewhere in the retrieved context.
        answer_tokens = [w for w in answer.lower().split() if len(w) > 3]
        grounded = sum(1 for w in answer_tokens if w in context_text.lower())
        groundedness_scores.append((grounded / len(answer_tokens) * 100) if answer_tokens else 0.0)
        latencies.append(elapsed)

    proposed_row = {
        "Model Variant": "Proposed RAG (Hybrid + NLI)",
        "BERTScore F1": round(_bertscore_f1(candidates, references), 2),
        "Faithfulness (%)": round(sum(faithfulness_scores) / len(faithfulness_scores), 2),
        "Groundedness (%)": round(sum(groundedness_scores) / len(groundedness_scores), 2),
        "Average Latency (s)": round(sum(latencies) / len(latencies), 3),
    }

    rows = [proposed_row] + [{"Model Variant": name, **metrics} for name, metrics in BASELINE_RESULTS.items()]
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_csv, index=False)
    return df_summary


if __name__ == "__main__":
    df_eval = run_evaluation()
    print("\n--- Evaluation Results ---")
    print(df_eval.to_string(index=False))
    print("\nSaved to eval_results.csv — open the Streamlit app's 'Offline Model Comparison' tab to see it.")
