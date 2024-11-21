import io

from data import iterator

import json_repair
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr


lines = open("../results/clf_perf.txt").readlines()
idx = [i for i, line in enumerate(lines) if "F1 Score" in line]
assert len(idx) == len(list(iterator())), f"{len(idx)} != {len(list(iterator()))}"

tables = ["\n".join(lines[i + 2 : i + 23]) for i in idx]
perfs = {}

for table, (name, _) in zip(tables, iterator()):
    cur_tbl = pd.read_csv(
        io.StringIO(table),
        sep=r"\s+",
        header=None,
        names=["", "Accuracy", "Balanced Accuracy", "ROC AUC", "F1 Score", "Time Taken", "normalized_regret"],
    )
    cur_tbl.sort_values(by="F1 Score", axis=0, ascending=False, inplace=True)

    perfs[name] = (cur_tbl.iloc[0, 0], cur_tbl.iloc[0, 4], np.mean(cur_tbl.iloc[:5, 4]), np.mean(cur_tbl.iloc[:, 4]))

with open("../results/logs.jsonl", "r") as f:
    lines = list(f)

complexity = [json_repair.loads(line) for line in lines]
measures = set([row["metric"] for row in complexity])

corr_df = [[measure] for measure in measures]
for i in range(1, len(perfs["defect:ivy"])):
    corr_data = {}
    for row in complexity:
        if row["metric"] not in corr_data:
            corr_data[row["metric"]] = []
        corr_data[row["metric"]].append((row["complexity"], perfs[row["dataset"]][i]))

    for j, metric in enumerate(corr_data):
        corr_df[j].append(round(pearsonr(*zip(*corr_data[metric])).statistic, 2))
        corr_df[j].append(round(spearmanr(*zip(*corr_data[metric])).statistic, 2))

print(
    pd.DataFrame(
        corr_df,
        columns=[
            "Measure",
            "Pearson: Best",
            "Spearman: Best",
            "Pearson: Mean5",
            "Spearman: Mean5",
            "Pearson: MeanAll",
            "Spearman: MeanAll",
        ],
    )
)
