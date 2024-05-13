import warnings

from data.bugzilla_issue import load_issue_lifetime_prediction_data
from metrics import CSGMetric


warnings.filterwarnings("ignore")
for filename in ["chromium", "eclipse", "firefox"]:
    for n_class in [2, 3, 5, 7, 9]:
        dataset = load_issue_lifetime_prediction_data(
            "../data/issue_close_time_prediction/bugzilla/", filename, n_class
        )

        metric = CSGMetric(dataset.x_train, dataset.y_train)
        csg = metric.get_complexity()

        print(f"{filename}-{n_class}class: {csg}")
