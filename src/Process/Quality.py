from Bio import SeqIO
import subprocess
import os
import tempfile
from sklearn.model_selection import train_test_split
import pandas as pd


def RunCDHIT(fasta_file, identity_threshold=0.7, **cdhit_options):
    """
    Run CD-HIT on the given FASTA file to remove redundant sequences.

    Parameters:
    fasta_file (str): Path to the input FASTA file.
    identity_threshold (float): Sequence identity threshold for clustering (default: 0.9).
    **cdhit_options: Additional single-value CD-HIT options without the leading dash.
        Example: T=8 passes -T 8.

    Returns:
    tuple: Representative sequences and their labels.
    """
    reserved_options = {"i", "o", "c", "n"}
    for option, value in cdhit_options.items():
        if not option.isascii() or len(option) != 1 or not option.isalpha():
            raise ValueError(f"Invalid CD-HIT option name: {option!r}")
        if option in reserved_options:
            raise ValueError(
                f"CD-HIT option -{option} is controlled by RunCDHIT"
            )
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise TypeError(
                f"Value for CD-HIT option -{option} must be a string or number"
            )

    with tempfile.TemporaryDirectory(prefix="cdhit_") as temp_dir:
        base_name = os.path.basename(fasta_file)
        output_file = os.path.join(temp_dir, base_name + "_cdhit.fasta")

        cmd = [
            "cd-hit",
            "-i", fasta_file,
            "-o", output_file,
            "-c", str(identity_threshold),
            "-n", "5",
        ]
        for option, value in cdhit_options.items():
            cmd.extend([f"-{option}", str(value)])

        subprocess.run(cmd, check=True)

        representative_sequences = []
        representative_ids = []
        for record in SeqIO.parse(output_file, "fasta"):
            representative_sequences.append(str(record.seq))
            representative_ids.append(record.id)

    labels = [record_id.split("|")[-1] for record_id in representative_ids]
    return representative_sequences, labels

def AlignSequencesWithBlast(
    database_sequences,
    query_sequences,
    max_target_seqs = 1,
    task="blastp-short",
    **blast_options,
):
    """Align query proteins against a temporary BLAST database.

    Additional blastp command-line options can be passed as keyword arguments,
    for example evalue=1e-5, word_size=3, seg="no", or num_threads=4.
    """
    if task not in {"blastp", "blastp-short"}:
        raise ValueError("task must be 'blastp' or 'blastp-short'")
    if not database_sequences:
        raise ValueError("database_sequences must contain at least one sequence")
    if not query_sequences:
        return []

    reserved_options = {"query", "db", "out", "outfmt", "task"}
    for option, value in blast_options.items():
        if not option.isascii() or not option.replace("_", "").isalnum():
            raise ValueError(f"Invalid BLAST option name: {option!r}")
        if option.lower() in reserved_options:
            raise ValueError(f"BLAST option -{option} is controlled by this function")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(
                f"Value for BLAST option -{option} must be a string, number, or bool"
            )

    with tempfile.TemporaryDirectory(prefix="blast_") as temp_dir:
        database_fasta = os.path.join(temp_dir, "database.fasta")
        query_fasta = os.path.join(temp_dir, "queries.fasta")
        database_prefix = os.path.join(temp_dir, "protein_db")
        output_file = os.path.join(temp_dir, "alignments.tsv")

        for fasta_file, sequences, id_prefix in (
            (database_fasta, database_sequences, "subject"),
            (query_fasta, query_sequences, "query"),
        ):
            with open(fasta_file, "w") as fasta:
                for index, sequence in enumerate(sequences):
                    sequence_text = "".join(str(sequence).split()).upper()
                    if not sequence_text:
                        raise ValueError(f"Sequence at index {index} is empty")
                    fasta.write(f">{id_prefix}_{index}\n{sequence_text}\n")

        make_db_cmd = [
            "makeblastdb",
            "-in", database_fasta,
            "-dbtype", "prot",
            "-parse_seqids",
            "-out", database_prefix,
        ]
        subprocess.run(make_db_cmd, check=True)

        outfmt = (
            "6 qseqid sseqid pident length mismatch gapopen qstart qend "
            "sstart send evalue bitscore qlen slen qseq sseq"
        )
        blast_cmd = [
            "blastp",
            "-query", query_fasta,
            "-db", database_prefix,
            "-task", task,
            "-max_target_seqs", str(max_target_seqs),
            "-max_hsps", "1",
            "-outfmt", outfmt,
            "-out", output_file,
        ]
        for option, value in blast_options.items():
            if isinstance(value, bool):
                value = "yes" if value else "no"
            blast_cmd.extend([f"-{option}", str(value)])

        subprocess.run(blast_cmd, check=True)

        hits = []
        with open(output_file) as results:
            for line in results:
                fields = line.rstrip("\n").split("\t")
                query_start, query_end = int(fields[6]), int(fields[7])
                subject_start, subject_end = int(fields[8]), int(fields[9])
                query_length, subject_length = int(fields[12]), int(fields[13])

                hits.append(
                    {
                        "query_index": int(fields[0].split("_")[-1]),
                        "database_index": int(fields[1].split("_")[-1]),
                        "identity_percent": float(fields[2]),
                        "alignment_length": int(fields[3]),
                        "mismatches": int(fields[4]),
                        "gap_opens": int(fields[5]),
                        "query_coverage_percent": 100 * (query_end - query_start + 1) / query_length,
                        "database_coverage_percent": 100 * (subject_end - subject_start + 1) / subject_length,
                        "evalue": float(fields[10]),
                        "bit_score": float(fields[11]),
                        "aligned_query": fields[14],
                        "aligned_database": fields[15],
                    }
                )

    return pd.DataFrame(hits)

def SplitData(sequences, labels, test_size=0.2, random_state=42):
    """
    Split the sequences and labels into training and testing sets.

    Parameters:
    sequences (list): List of sequences.
    labels (list): List of corresponding labels.
    test_size (float): Proportion of the dataset to include in the test split (default: 0.2).
    random_state (int): Random seed for reproducibility (default: 42).

    Returns:
    tuple: Training sequences, testing sequences, training labels, testing labels.
    """
    return train_test_split(sequences, labels, test_size=test_size, random_state=random_state, stratify=labels)