
import pandas as pd
from Bio import SeqIO
import subprocess

def Seq2Tab(Path):
    """Converts a FASTA file to a pandas DataFrame.
    Args:
        Path (str): Path to the FASTA file.
    Returns:
        pd.DataFrame: A DataFrame containing the sequence IDs and sequences.
    """
    data = {
        "Access":   [str(record.id) for record in SeqIO.parse(Path, 'fasta')],
        "Sequence": [str(record.seq) for record in SeqIO.parse(Path, 'fasta')],
        "Label":    [str(record.id).split('|')[-1] for record in SeqIO.parse(Path, 'fasta')],
        "Length":   [len(str(record.seq)) for record in SeqIO.parse(Path, 'fasta')],
    }
    return pd.DataFrame(data)

def Tab2Seq(Data,Output):
    """Converts a pandas DataFrame to a FASTA file.
    Args:
        Data (pd.DataFrame): A DataFrame containing the sequence IDs and sequences.
        Output (str): Path to the output FASTA file.
    """
    with open(Output, 'w+') as f:
        for index, row in Data.iterrows():
            f.write(f">{row['Access']}|{row['Label']}\n{row['Sequence']}\n")

def RunCDHIT(input, output, max_identity=0.95, min_cov = 0.8, size_diff=0.8):
    """Runs CD-HIT to cluster sequences in a FASTA file based on sequence identity.
    Args:        input (str): Path to the input FASTA file.
        output (str): Path to the output FASTA file where the clustered sequences will be saved.
        max_identity (float, optional): The maximum sequence identity threshold for clustering. Defaults to 0.95.
        min_cov (float, optional): The minimum coverage threshold for clustering. Defaults to 0.8."""
    command = [
        "cd-hit",
        "-i", input,
        "-o", output,
        "-c", str(max_identity),
        "-T", str(12),
        "-p", str(1),
        "-sc", str(1),
        "-sf", str(1),
        "-aL", str(min_cov),    
        "-aS", str(min_cov),    
        "-s",  str(size_diff)
        ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

def CountId(path):
    """Counts the number of sequences in a FASTA file.
    Args:
        path (str): Path to the FASTA file.
    Returns:
        int: The number of sequences in the FASTA file.
    """
    Command = ["grep", "-c", ">", path]
    result = subprocess.run(Command, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())

def ClusterCounter(path):
    """Runs CD-HIT at different identity thresholds and counts the number of clusters formed at each threshold.
    Args:
        path (str): Path to the input FASTA file.
    Returns:
        tuple: A tuple containing two lists: the first list contains the number of clusters at each threshold, and the second list contains the corresponding percentages of the total number of sequences.
    """
    TotalSize = CountId(path)
    counts = [TotalSize, ]
    perc = [100.00]
    for i in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]:
        RunCDHIT(
            input = path,
            output = f"{path}.out",
            max_identity = i
        )
        PartialSize = CountId(f"{path}.out")
        Percentage = (PartialSize / TotalSize) * 100
        counts.append(PartialSize)
        perc.append(Percentage)
        subprocess.run(["rm", f"{path}.out"], check=True)
        subprocess.run(["rm", f"{path}.out.clstr"], check=True)
    return counts, np.round(perc,2)
