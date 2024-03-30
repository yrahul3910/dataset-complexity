import warnings

from itertools import product
from pathlib import Path

from metrics import CSGMetric

import numpy as np

from raise_utils.data import Data, DataLoader
from raise_utils.hooks import Hook
from raise_utils.transforms import Transform
from sklearn.manifold import TSNE


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

    base_path = str((Path(__file__).parent.parent / "data/defect_prediction/").resolve())
    data = DataLoader.from_files(
        base_path=base_path,
        files=defect_file_dic[dataset],
        hooks=[Hook("binarize", _binarize)],
    )
    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)

    transform = Transform("normalize")
    transform.apply(data)

    return data


warnings.filterwarnings("ignore")
for filename, tsne in product(defect_file_dic, [True, False]):
    dataset = load_defect_data(filename)
    if tsne:
        x_train = TSNE(n_components=3).fit_transform(dataset.x_train)
    else:
        x_train = dataset.x_train

    metric = CSGMetric(dataset.x_train, dataset.y_train)
    csg = metric.get_complexity()
    print(f"{filename} (TSNE: {tsne}): {csg}")
    # make_graph(estimator.difference, title=filename, classes=["clean", "defective"])
