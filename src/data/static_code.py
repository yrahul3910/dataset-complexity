import numpy as np
import pandas as pd

from raise_utils.data import Data
from raise_utils.transforms import Transform
from sklearn.model_selection import train_test_split


datasets = ["ant", "cassandra", "commons", "derby", "jmeter", "lucene-solr", "tomcat"]


def static_code_loader():
    for dataset in datasets:
        yield load_static_code_data("../data/static_code_warnings/", dataset)


def load_static_code_data(base_path: str, dataset: str) -> tuple[str, Data]:
    train_file = f"{base_path}/train/{dataset}_B_features.csv"
    test_file = f"{base_path}/test/{dataset}_C_features.csv"

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    df = pd.concat((train_df, test_df), join="inner")

    X = df.drop("category", axis=1)
    y = df["category"]

    y[y == "close"] = 1
    y[y == "open"] = 0

    y = np.array(y, dtype=np.float32)

    X = X.select_dtypes(exclude=["object"]).astype(np.float32)

    data = Data(*train_test_split(X, y, test_size=0.2, shuffle=False))
    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)

    transform = Transform("normalize")
    transform.apply(data)

    return f"staticcode:{dataset}", data
