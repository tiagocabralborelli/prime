import re
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from google.colab import files

# ===== 上传文件 =====
FASTA = next(iter(files.upload()))

CLSTR = next(iter(files.upload()))

SEED = 42

# 1. 读取全部序列，过滤不足5 aa的记录
seqs = {}
sid, parts = None, []

def save():
    if sid is None:
        return
    seq = "".join("".join(parts).split()).upper()
    if len(seq) < 5:
        return
    if sid in seqs:
        raise ValueError(f"重复ID：{sid}")
    if sid.rsplit("|", 1)[-1] not in {"label=0", "label=1"}:
        raise ValueError(f"ID末尾必须为 |label=0 或 |label=1：{sid}")
    seqs[sid] = seq

with open(FASTA, encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            save()
            sid, parts = line[1:].split()[0], []
        else:
            parts.append(line)
save()

# 2. 读取聚类归属，检查两个文件是否对应
group_map = {}
with open(CLSTR, encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">Cluster"):
            cluster = line
        else:
            member = re.search(r">(.+)\.\.\.", line).group(1).split()[0]
            if member in group_map:
                raise ValueError(f".clstr中重复ID：{member}")
            group_map[member] = cluster

# 先显示两个文件的ID差异
only_fasta = set(seqs) - set(group_map)
only_clstr = set(group_map) - set(seqs)

print("仅在FASTA中的ID数量：", len(only_fasta))
print("示例：", sorted(only_fasta)[:5])

print("仅在.clstr中的ID数量：", len(only_clstr))
print("示例：", sorted(only_clstr)[:5])

# 再检查是否对应
# .clstr中有FASTA未提供的成员时，仍停止检查
if only_clstr:
    raise ValueError(".clstr中存在FASTA未提供的成员")

# 排除FASTA中没有聚类归属的记录
for sid in sorted(only_fasta):
    print(f"排除未参与聚类的序列：{sid}")
    del seqs[sid]

ids = list(seqs)
labels = np.array([
    int(s.rsplit("|", 1)[-1].split("=")[1])
    for s in ids
])
groups = np.array([group_map[s] for s in ids])

if len(set(labels)) != 2 or len(set(groups)) < 5:
    raise ValueError("需要同时有正负样本，并且至少有5个聚类")


# 3. 分层分组划分：5折中选1折作为约20%的测试集
# 根据数量与标签比例选择
best = None
totals = np.bincount(labels, minlength=2)

for seed in range(SEED, SEED + 20):
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed
    )
    for train_idx, test_idx in splitter.split(ids, labels, groups):
        if len(set(labels[train_idx])) < 2 or \
           len(set(labels[test_idx])) < 2:
            continue

        test_counts = np.bincount(labels[test_idx], minlength=2)
        fractions = np.r_[len(test_idx) / len(ids), test_counts / totals]
        score = np.mean((fractions - 0.2) ** 2)

        if best is None or score < best[0]:
            best = (score, train_idx, test_idx)

if best is None:
    raise ValueError("无法得到两边均含正负样本的整簇划分")

_, train_idx, test_idx = best
assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))

# 4. 保存，并输出实际数量及标签比例
for name, indices in [("train", train_idx), ("test", test_idx)]:
    counts = Counter(labels[indices])
    with open(f"/content/{name}.fasta", "w") as f:
        for i in indices:
            f.write(f">{ids[i]}\n{seqs[ids[i]]}\n")

    print(
        f"{name}：{len(indices)}条（{len(indices)/len(ids):.1%}），"
        f"阳性={counts[1]}，阴性={counts[0]}，"
        f"阳性占比={counts[1]/len(indices):.1%}"
    )