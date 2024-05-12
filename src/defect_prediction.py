import warnings

from itertools import product

from data.defect import defect_file_dic, load_defect_data

from metrics import CSGMetric
from sklearn.manifold import TSNE


warnings.filterwarnings("ignore")
for filename, tsne in product(defect_file_dic, [True, False]):
    dataset = load_defect_data('../data/', filename)
    if tsne:
        x_train = TSNE(n_components=3).fit_transform(dataset.x_train)
    else:
        x_train = dataset.x_train

    metric = CSGMetric(dataset.x_train, dataset.y_train)
    csg = metric.get_complexity()
    print(f"{filename} (TSNE: {tsne}): {csg}")
    # make_graph(estimator.difference, title=filename, classes=["clean", "defective"])
