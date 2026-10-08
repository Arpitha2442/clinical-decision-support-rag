import sys
import os
import time

def main():
    print("=" * 70)
    print(" 🎓 PharmaRAG Conference Benchmark & Results Reproducibility Script")
    print("=" * 70)
    
    t_start = time.time()
    
    # 1. Run evaluation suite
    print("\n[Step 1/3] Executing 36-Query Clinical Evaluation Suite...")
    from evaluate import run_evaluation
    df_eval = run_evaluation(
        output_summary_csv="eval_results.csv",
        output_per_query_csv="eval_per_query.csv"
    )
    print("  -> Exported eval_results.csv & eval_per_query.csv")
    print("  -> Exported eval_table.tex (IEEE/ACM LaTeX table)")
    
    # 2. Generate 300 DPI figures
    print("\n[Step 2/3] Generating High-Resolution 300 DPI Conference Figures...")
    from plot_metrics import generate_performance_plots
    generate_performance_plots(
        summary_csv="eval_results.csv",
        per_query_csv="eval_per_query.csv",
        output_img="rag_performance_metrics.png",
        conference_img="conference_eval_plots.png"
    )
    print("  -> Exported rag_performance_metrics.png")
    print("  -> Exported conference_eval_plots.png")
    
    # 3. Summary output
    t_elapsed = time.time() - t_start
    print("\n[Step 3/3] Summary of Results Across Ablation Variants:")
    print("-" * 70)
    print(df_eval.to_string(index=False))
    print("-" * 70)
    print(f"\n[OK] Reproducibility Pipeline Finished Successfully in {t_elapsed:.2f} seconds!")

if __name__ == "__main__":
    main()
