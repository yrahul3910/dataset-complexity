import glob

import numpy as np

from raise_utils.data import Data, DataLoader
from raise_utils.transforms import Transform


def uci_loader():
    for dataset in glob.glob("../data/uci/*.csv"):
        filename = dataset.split("/")[-1].replace(".csv", "")
        yield load_uci_data("../data/uci", filename)


def load_uci_data(base_path: str, dataset: str) -> tuple[str, Data]:
    data = DataLoader.from_file(f"{base_path}/{dataset}.csv", col_start=0, header=None, target=-1)

    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)

    transform = Transform("normalize")
    transform.apply(data)

    return f"uci:{dataset}", data
