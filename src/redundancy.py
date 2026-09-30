import subprocess

def run_cdhit(fasta_path, identity_threshold, output_path):
    cmd = [
        "cd-hit",
        "-i", fasta_path,
        "-o", output_path,
        "-c", str(identity_threshold),
        "-aS","0.8",  #默认无限制
        "-g", "1",   # 精确模式
        "-d", "0",   # 仅保留序列 ID
        "-l", "4",  # 最小序列长度（按需调整）
        "-M", "0",   # 不限内存
        "-T", "4",   # 线程数（按机器调整）
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_cdhit("input.fasta", 0.8, "output.fasta")