import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def create_heatmap(data, ranks, learning_rates, title="Heatmap Title", output_file="heatmap.png"):
    """
    Create a heatmap with a 3x3 grid and save it to a file.

    Parameters:
    - data: 2D numpy array of shape (3, 3) with percentages to display.
    - ranks: List of rank labels for the x-axis.
    - learning_rates: List of learning rate labels for the y-axis.
    - title: Title for the heatmap.
    - output_file: File name to save the heatmap (e.g., "heatmap.png").
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = mcolors.LinearSegmentedColormap.from_list("vibrant", ["#FF6F61", "#FFD700", "#ADFF2F", "#32CD32", "#1E90FF"])
    
    # Normalize data between 0 and 100
    norm = mcolors.Normalize(vmin=50, vmax=100)

    # Plot the heatmap
    c = ax.imshow(data, cmap=cmap, norm=norm)

    # Add the percentages to each cell
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j, i, f"{data[i, j]:.0f}%", ha="center", va="center", color="black"
            )

    # Set axis labels and ticks
    ax.set_xticks(np.arange(len(ranks)))
    ax.set_yticks(np.arange(len(learning_rates)))
    ax.set_xticklabels(ranks)
    ax.set_yticklabels(learning_rates)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Learning Rate")

    # Set the title with bold font and higher position
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Add a colorbar with a new label
    cbar = plt.colorbar(c, ax=ax, orientation="vertical", label="Average Extraction Accuracy (%)")
    cbar.set_label("Average Extraction Accuracy", fontsize=12, fontweight='bold')

    # Save the heatmap to a file
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Heatmap saved to {output_file}")

    plt.show()


# Example usage
if __name__ == "__main__":
    # Example data (3x3 grid)
    data = np.array([
        [72, 92, 88],
        [94, 98, 96],
        [91, 93, 94]
    ])

    # Example rank and learning rate labels
    ranks = ["4", "8", "16"]
    learning_rates = ["2e-4", "5e-4", "8e-4"]

    # Heatmap title and output file name
    title = "LoRA Accuracy by Hyperparameters"
    output_file = "performance_heatmap.png"

    create_heatmap(data, ranks, learning_rates, title, output_file)
