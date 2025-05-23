import pandas as pd
import matplotlib.pyplot as plt
import os

# ── Config ─────────────────────────────────────────────────────────
CSV_PATH = "detoxification_evaluation_all.csv"
OUTPUT_CSV = "detox_comparison_report.csv"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# ── Group by model ─────────────────────────────────────────────────
metrics = [
    "toxicity_reference",
    "toxicity_generated",
    "similarity_original_reference",
    "similarity_original_generated"
]

summary_df = df.groupby("model_id")[metrics].mean()
summary_df["toxicity_reduction"] = summary_df["toxicity_reference"] - summary_df["toxicity_generated"]
summary_df["similarity_drop"] = summary_df["similarity_original_reference"] - summary_df["similarity_original_generated"]
summary_df = summary_df.sort_values("toxicity_reduction", ascending=False)

# ── Save summary to CSV ────────────────────────────────────────────
summary_df.to_csv(OUTPUT_CSV)
print(f"✅ Saved comparison report to: {OUTPUT_CSV}")

# ── Plotting function ──────────────────────────────────────────────
def plot_metric(metric, title, ylabel, output_file):
    plt.figure(figsize=(10, 5))
    summary_df[metric].plot(kind="bar", rot=45)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, output_file))
    plt.close()
    print(f"📊 Saved plot: {output_file}")

# ── Generate plots ────────────────────────────────────────────────
plot_metric("toxicity_reference", "Average Reference Toxicity", "Toxicity", "toxicity_reference.png")
plot_metric("toxicity_generated", "Average Generated Toxicity", "Toxicity", "toxicity_generated.png")
plot_metric("toxicity_reduction", "Toxicity Reduction (Ref - Gen)", "Reduction", "toxicity_reduction.png")

plot_metric("similarity_original_reference", "Original ↔ Reference Similarity", "Cosine Similarity", "similarity_reference.png")
plot_metric("similarity_original_generated", "Original ↔ Generated Similarity", "Cosine Similarity", "similarity_generated.png")
plot_metric("similarity_drop", "Similarity Drop (Ref - Gen)", "Drop", "similarity_drop.png")

print(f"\n📁 All plots saved to: {PLOT_DIR}/")
