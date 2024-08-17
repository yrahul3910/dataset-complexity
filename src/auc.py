import json_repair
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler


def calculate_auc(group):
    if group["ground_truth"].nunique() < 2:
        return np.nan  # AUC-ROC cannot be calculated if there is only one class in the group
    return roc_auc_score(group["ground_truth"], group["normalized_value"])


ground_truth_df = pd.read_csv("../results/ground_truth.csv")
ground_truth_df["complex"] = [0 if row["Simple/Complex"] == "Simple" else 1 for _, row in ground_truth_df.iterrows()]
ground_truth_map = {row["Dataset"]: row["complex"] for _, row in ground_truth_df.iterrows()}

with open("../results/logs.jsonl", "r") as f:
    lines = list(f)

results_df = pd.DataFrame([json_repair.loads(line) for line in lines])
results_df["ground_truth"] = results_df["dataset"].map(ground_truth_map)

scaler = MinMaxScaler()
results_df["normalized_value"] = results_df.groupby("metric")["complexity"].transform(
    lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
)
auc_scores = results_df.groupby("metric").apply(calculate_auc)

auc_scores.sort_values(ascending=False, inplace=True)
auc_scores.to_csv("../results/auc_scores.csv")
print(auc_scores)
