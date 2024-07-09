from data.bugzilla_issue import bugzilla_loader
from data.defect import defect_loader
from data.github_issue import github_issue_loader
from data.static_code import static_code_loader
from data.uci import uci_loader

from loguru import logger
from metrics import get_all_estimators


def iterator():
    data_loaders = [defect_loader, bugzilla_loader, github_issue_loader, static_code_loader, uci_loader]

    for loader in data_loaders:
        yield from loader()


logger.add("logs.txt")
for name, dataset in iterator():
    for metric in get_all_estimators():
        estimator = metric(dataset.x_train, dataset.y_train)
        complexity = estimator.get_complexity()

        # Log as JSONL
        logger.info(
            {
                "dataset": name,
                "metric": estimator.__name__,
                "complexity": complexity,
            }
        )
