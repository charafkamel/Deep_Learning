import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import os
import pandas as pd
import matplotlib.pyplot as plt

def create_histogram():
    """
    Create a histogram of model performance based on combined score, similarity, and toxicity.
    """
    # Ensure the output directory exists
    os.makedirs("../visualizations", exist_ok=True)
    # Load the full dataset
    print(os.getcwd())
    print("!!\n\n!!!")
    df = pd.read_csv("eval/results/detoxification_evaluation_merged.csv")

    # Count number of samples per model
    counts = df.groupby("model_id").size().reset_index(name="n")

    # Group by model_id and compute mean and std
    agg_df = df.groupby("model_id").agg({
        "toxicity_generated": ["mean", "std"],
        "similarity_original_generated": ["mean", "std"]
    }).reset_index()

    # Flatten columns
    agg_df.columns = [
        "model_id",
        "toxicity_generated", "std_toxicity_generated",
        "similarity_original_generated", "std_similarity_original_generated"
    ]

    # Merge with sample counts
    df = agg_df.merge(counts, on="model_id")

    # Drop models with invalid names
    df = df[~df['model_id'].str.startswith('_')]

    # Compute combined score
    df['combined_score'] = 0.5 * df['similarity_original_generated'] + 0.5 * (1 - df['toxicity_generated'])

    # Compute standard error of the mean for each
    df['se_sim'] = df['std_similarity_original_generated'] / np.sqrt(df['n'])
    df['se_tox'] = df['std_toxicity_generated'] / np.sqrt(df['n'])

    # Compute standard error of the combined score (assuming independence)
    df['se_combined'] = 0.5 * df['se_sim'] + 0.5 * df['se_tox']

    # Sort by combined score
    df = df.sort_values(by='combined_score', ascending=False).reset_index(drop=True)

    # Extract values
    models = df["model_id"]
    similarity = df["similarity_original_generated"]
    toxicity = df["toxicity_generated"]
    combined_score = df["combined_score"]
    std_sim = df["se_sim"]
    std_tox = df["se_tox"]
    std_combined = df["se_combined"]

    # Set up positions
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(12, 8), dpi=1000)

    # Plot main combined score with error bars
    ax.barh(x, combined_score, height=0.6, xerr=std_combined, color='mediumseagreen', label='Combined Score', capsize=4)

    # Add similarity and toxicity with scaled SE error bars
    ax.barh(x - 0.2, similarity, height=0.2, color='steelblue', label='Similarity', capsize=3)
    ax.barh(x + 0.2, toxicity, height=0.2, color='darkorange', label='Toxicity', capsize=3)

    # Plot aesthetics
    ax.set_xlabel('Score')
    ax.set_title('Model Performance: Combined Score vs. Similarity & Toxicity')
    ax.set_yticks(x)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    output_path = "visualizations/Histogram.png"
    print("?!?!?!\nSaving histogram to:", output_path)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def create_table():
    """
    Create a table summarizing model performance metrics.
    """
    # Ensure the output directory exists
    os.makedirs("visualizations", exist_ok=True)
    # Load CSV
    df = pd.read_csv("eval/results/detoxification_evaluation_merged.csv")

    # Group and aggregate
    agg = df.groupby("model_id").agg(
        similarity_mean=("similarity_original_generated", "mean"),
        similarity_std=("similarity_original_generated", "std"),
        toxicity_mean=("toxicity_generated", "mean"),
        toxicity_std=("toxicity_generated", "std"),
        n=("similarity_original_generated", "count")
    ).reset_index()

    # Compute combined score
    agg["combined_score"] = 0.5 * agg["similarity_mean"] + 0.5 * (1 - agg["toxicity_mean"])
    agg = agg.sort_values("combined_score", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1

    # Format for display
    agg["Similarity (mean ± std)"] = agg.apply(
        lambda row: f"{row['similarity_mean']:.2f} ± {row['similarity_std']:.2f}", axis=1)
    agg["Toxicity (mean ± std)"] = agg.apply(
        lambda row: f"{row['toxicity_mean']:.2f} ± {row['toxicity_std']:.2f}", axis=1)
    agg["Combined Score"] = agg["combined_score"].round(3)

    # Final summary
    summary_df = agg[[
        "model_id", "Similarity (mean ± std)", "Toxicity (mean ± std)", "Combined Score"
    ]].rename(columns={"model_id": "Model"})

    # Create output folder
    os.makedirs("../visualizations", exist_ok=True)

    # Plot setup
    fig, ax = plt.subplots(figsize=(20, len(summary_df) * 0.6))
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
    )

    # Adjust font and scale
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.5, 1.8)  # wider for long strings, taller for clarity

    # Optional: set specific column widths
    col_widths = [0.45, 0.2, 0.2, 0.15]  # tune if needed
    for i, width in enumerate(col_widths):
        for j in range(len(summary_df) + 1):  # +1 includes header
            cell = table[(j, i)]
            cell.set_width(width)

    # Save the figure
    plt.savefig("../visualizations/table.png", bbox_inches="tight", dpi=300)
    plt.close()

def main():
    """
    Main function to create visualizations.
    """
    create_histogram()
    create_table()

if __name__ == "__main__":
    main()
    