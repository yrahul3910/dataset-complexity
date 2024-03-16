import warnings

import numpy as np

from raise_utils.data import Data, DataLoader
from raise_utils.hooks import Hook
from spectral_metric.estimator import CumulativeGradientEstimator


# Dataset filenames
defect_file_dic = {
    "ivy": ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],
    "lucene": ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],
    "poi": ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],
    "synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],
    "velocity": ["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"],
    "camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"],
    "jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"],
    "log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"],
    "xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"],
    "xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"],
}


def load_defect_data(dataset: str) -> Data:
    def _binarize(x, y):
        y[y > 1] = 1

    base_path = "../../data/defect_prediction/"
    data = DataLoader.from_files(
        base_path=base_path,
        files=defect_file_dic[dataset],
        hooks=[Hook("binarize", _binarize)],
    )
    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)

    return data


warnings.filterwarnings("ignore")
for filename in defect_file_dic:
    dataset = load_defect_data(filename)
    estimator = CumulativeGradientEstimator(M_sample=250, k_nearest=5)
    estimator.fit(data=dataset.x_train, target=dataset.y_train)

    print(f"{filename}: {estimator.csg}")
    # make_graph(estimator.difference, title=filename, classes=["clean", "defective"])
