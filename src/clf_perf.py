import pickle

from data import iterator

import pandas as pd

from lazypredict.Supervised import CLASSIFIERS, LazyClassifier


results = {}
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
for name, dataset in iterator():
    print(name)
    highmem_classifiers = ["LabelSpreading","LabelPropagation","BernoulliNB","KNeighborsClassifier",
                           "ElasticNetClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier"]

    # Remove the high memory classifiers from the list
    classifiers = [c for c in CLASSIFIERS if c[0] not in highmem_classifiers]
    clf = LazyClassifier(classifiers=classifiers)
    scores, _ = clf.fit(dataset.x_train, dataset.x_test, dataset.y_train, dataset.y_test)

    # Compute noramlized regret for F1
    scores['normalized_regret'] = 1 - scores['F1 Score'] / scores['F1 Score'].max()
    results[name] = scores

    print(scores)

with open('clf_perf.pkl', 'wb') as f:
    pickle.dump(results, f)
