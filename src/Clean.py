import click
from Process.Quality import RunCDHIT, SplitData, AlignSequencesWithBlast
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import yaml


config_path = Path(__file__).with_name("Params.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

clean_config = config["clean"]

@click.command()
@click.option('--file', help='Path to the file to clean', required=True)
@click.option('--output', help='Path to the output file', required=True)
@click.option('--name', help='Name of the output file', required=True)

def main(file, output, name):
    """
    Clean the input file and save the cleaned data to the output file.
    """

    os.makedirs(output, exist_ok=True)
 

    Sequences,Labels = RunCDHIT(
        file, 
        identity_threshold=clean_config["cdhit"]["id_threshold"])
    TrainSequences, TestSequences, TrainLabels, TestLabels = SplitData(Sequences, Labels, test_size=clean_config["split"]["test_size"], random_state=clean_config["split"]["random_state"])
    BlastHits = AlignSequencesWithBlast(
        TrainSequences,
        TestSequences,
        task=clean_config["blast"]["task"],
        evalue=clean_config["blast"]["evalue"],
        word_size=clean_config["blast"]["word_size"],
        seg=clean_config["blast"]["seg"],
        num_threads=clean_config["blast"]["num_threads"],
        max_target_seqs=clean_config["blast"]["max_target_seqs"],
    )


    with open(os.path.join(output,f"{name}-test.fasta"), "w+") as testfile:
        for seq, label in zip(TestSequences, TestLabels):
            testfile.write(f">{label}\n{seq}\n")

    with open(os.path.join(output,f"{name}-train.fasta"), "w+") as trainfile:
        for seq, label in zip(TrainSequences, TrainLabels):
            trainfile.write(f">{label}\n{seq}\n")


    #Print similarity and coverage distribution
    plot_data = BlastHits[
        ["identity_percent", "query_coverage_percent", "bit_score", "evalue"]
    ].dropna()
    plot_data = plot_data[
        np.isfinite(plot_data["evalue"]) & (plot_data["evalue"] > 0)
    ]

    # Scale bit scores to readable marker sizes.
    scores = plot_data["bit_score"]
    score_range = scores.max() - scores.min()
    if score_range == 0:
        sizes = np.full(len(scores), 110.0)
    else:
        sizes = 20 + 180 * (scores - scores.min()) / score_range

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        plot_data["query_coverage_percent"],
        plot_data["identity_percent"],
        s=sizes,
        c=np.log10(plot_data["evalue"]),
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.3,
    )

    ax.set_ylabel("Identity (%)")
    ax.set_xlabel("Query coverage (%)")
    ax.set_title(f"BLAST hit identity: {name} test set coverage")

    legend_scores = np.unique(np.quantile(scores, [0, 0.5, 1]))
    legend_handles = []
    for score in legend_scores:
        marker_size = 110.0 if score_range == 0 else 20 + 180 * (score - scores.min()) / score_range
        handle = ax.scatter(
            [],
            [],
            s=marker_size,
            color="gray",
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3,
            label=f"{score:.1f}",
        )
        legend_handles.append(handle)
    ax.legend(handles=legend_handles, title="Bit score (marker size)")

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("log10(E-value)")

    plt.tight_layout()
    plt.savefig(os.path.join(output, f"{name}_blast_hits.png"), dpi=300)
    plt.show()






if __name__ == "__main__":
    main()