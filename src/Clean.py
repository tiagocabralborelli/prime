import os
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml

from Process.Quality import AlignSequencesWithBlast, RunCDHIT, SplitData

config_path = Path(__file__).with_name("Params.yaml")
with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

clean_config = config["clean"]


@click.command(help="Clean input FASTA file, perform redundancy reduction and train, test and houlout splits.")
@click.option("--file", "-f", help="Path to the input FASTA file to clean.", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", help="Path to the output directory.", required=True, type=click.Path(file_okay=False))
@click.option("--name", "-n", help="Dataset/project prefix name.", required=True)
def main(file, output, name):
    """
    Clean the input file and save the cleaned data to the output directory.
    """
    os.makedirs(output, exist_ok=True)

    click.echo(f"Starting dataset preprocessing for: {name}")
    click.echo(f"• Input FASTA: {file}")
    click.echo(f"• Output directory: {output}")

    id_thresh = clean_config["cdhit"]["id_threshold"]
    click.echo(f"Running CD-HIT redundancy reduction (identity threshold: {id_thresh})...")
    sequences, labels = RunCDHIT(file, identity_threshold=id_thresh)
    click.echo(f"• Non-redundant sequences: {len(sequences)}")

    test_size = clean_config["split"]["test_size"]
    random_state = clean_config["split"]["random_state"]
    click.echo(f"Splitting dataset (test_size={test_size}, random_state={random_state})...")
    train_seqs, test_seqs, train_labels, test_labels = SplitData(
        sequences, labels, test_size=test_size, random_state=random_state
    )
    click.echo(f"• Training set size: {len(train_seqs)}")
    click.echo(f"• Test set size: {len(test_seqs)}")

    click.echo("Aligning train vs test sequences with BLAST...")
    blast_hits = AlignSequencesWithBlast(
        train_seqs,
        test_seqs,
        task=clean_config["blast"]["task"],
        evalue=clean_config["blast"]["evalue"],
        word_size=clean_config["blast"]["word_size"],
        seg=clean_config["blast"]["seg"],
        num_threads=clean_config["blast"]["num_threads"],
        max_target_seqs=clean_config["blast"]["max_target_seqs"],
    )

    test_path = os.path.join(output, f"{name}-test.fasta")
    with open(test_path, "w", encoding="utf-8") as testfile:
        for seq, label in zip(test_seqs, test_labels):
            testfile.write(f">{label}\n{seq}\n")
    click.echo(f"Saved test FASTA: {test_path}")

    train_path = os.path.join(output, f"{name}-train.fasta")
    with open(train_path, "w", encoding="utf-8") as trainfile:
        for seq, label in zip(train_seqs, train_labels):
            trainfile.write(f">{label}\n{seq}\n")
    click.echo(f"Saved train FASTA: {train_path}")

    # Plot similarity and coverage distribution
    plot_data = blast_hits[
        ["identity_percent", "query_coverage_percent", "bit_score", "evalue"]
    ].dropna()
    plot_data = plot_data[
        np.isfinite(plot_data["evalue"]) & (plot_data["evalue"] > 0)
    ]

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
    ax.set_title(f"BLAST Hit Identity: {name} Test Set Coverage")

    legend_scores = np.unique(np.quantile(scores, [0, 0.5, 1]))
    legend_handles = []
    for score in legend_scores:
        marker_size = (
            110.0 if score_range == 0 else 20 + 180 * (score - scores.min()) / score_range
        )
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
    plot_path = os.path.join(output, f"{name}_blast_hits.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    click.echo(f"Saved BLAST plot: {plot_path}")
    click.echo(f"Preprocessing completed successfully for '{name}'.")


if __name__ == "__main__":
    main()