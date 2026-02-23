"""
step5_plot_comparison.py
Generates the final PPL comparison plot from Step 4 cumulative evaluation CSV results.
output :
./output/
`-- final_ppl_comparison_plot.png   (default; or <output_plot>)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def main(args):
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_path}")
        print("Please run step4_evaluate_cumulative.py first.")
        return


    sns.set_theme(style="whitegrid")



    plt.figure(figsize=(16, 8))

    ax = sns.lineplot(
        data=df,
        x="restored_count",
        y="ppl",
        hue="metric",
        style="metric",
        markers=True,
        dashes=False,
    )

    ax.set_title("PPL vs. Number of Restored Groups by Importance Metric", fontsize=16)
    ax.set_xlabel("Number of Restored Groups", fontsize=12)
    ax.set_ylabel("Perplexity (PPL)", fontsize=12)
    ax.legend(title="Importance Metric")
    ax.grid(True, which="both", linestyle="--")




    plt.tight_layout(pad=1.5)


    plt.savefig(args.output_plot, dpi=300)
    print(f"✅ Final PPL comparison plot saved to {args.output_plot}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 5: Create a final PPL comparison plot."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the cumulative_results.csv file from Step 4.",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="./output/final_ppl_comparison_plot.png",
        help="Path to save the final PPL comparison plot.",
    )
    args = parser.parse_args()
    main(args)
