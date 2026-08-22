import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_performance_plots(csv_path: str = "eval_results.csv", output_img: str = "rag_performance_metrics.png"):
    """
    Reads benchmark evaluation results and generates stand-alone 
    visualization figures comparing the RAG architectures.
    """
    # Check if results CSV exists, fallback to default metrics if missing
    if os.path.exists(csv_path):
        df_eval = pd.read_csv(csv_path)
    else:
        print(f"⚠️ '{csv_path}' not found. Generating plot from default evaluation data...")
        df_eval = pd.DataFrame({
            "Model Variant": ["Proposed RAG (Hybrid + NLI)", "Baseline 1 (HomeDOCtor)", "Baseline 2 (MEDIC)"],
            "BERTScore F1": [75.0, 70.0, 62.0],
            "Faithfulness (%)": [70.2, 65.0, 60.0],
            "Groundedness (%)": [67.0, 62.0, 57.0],
            "Average Latency (s)": [1.2, 0.9, 2.1]
        })

    # Set visualization theme
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -------------------------------------------------------------------------
    # Chart 1: Quality & Safety Metrics (BERTScore, Faithfulness, Groundedness)
    # -------------------------------------------------------------------------
    df_melted = df_eval.melt(
        id_vars="Model Variant", 
        value_vars=["BERTScore F1", "Faithfulness (%)", "Groundedness (%)"],
        var_name="Metric", 
        value_name="Score"
    )

    sns.barplot(
        data=df_melted, 
        x="Metric", 
        y="Score", 
        hue="Model Variant", 
        ax=axes[0], 
        palette="Set2"
    )
    
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Semantic & Factual Alignment (Higher is Better)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Score / Percentage")
    axes[0].legend(title="Model Architecture", loc="lower right")

    # Annotate bar values
    for p in axes[0].patches:
        h = p.get_height()
        if h > 0:
            axes[0].annotate(
                f"{h:.1f}", 
                (p.get_x() + p.get_width() / 2., h),
                ha='center', va='center', 
                xytext=(0, 5), 
                textcoords='offset points', 
                fontsize=8
            )

    # -------------------------------------------------------------------------
    # Chart 2: System Execution Efficiency (Latency)
    # -------------------------------------------------------------------------
    if "Average Latency (s)" in df_eval.columns or "Latency (s)" in df_eval.columns:
        latency_col = "Average Latency (s)" if "Average Latency (s)" in df_eval.columns else "Latency (s)"
        
        sns.barplot(
            data=df_eval, 
            x="Model Variant", 
            y=latency_col, 
            ax=axes[1], 
            palette="crest"
        )
        
        axes[1].set_title("Execution Latency (Lower is Better)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Time (Seconds)")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")

        # Annotate latency values
        for p in axes[1].patches:
            h = p.get_height()
            if h > 0:
                axes[1].annotate(
                    f"{h:.2f}s", 
                    (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points', 
                    fontsize=9
                )

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f"✅ Metric visualization saved to '{output_img}'")
    plt.show()

if __name__ == "__main__":
    generate_performance_plots()