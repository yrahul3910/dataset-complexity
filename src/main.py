import gc

from metrics import get_all_estimators

from .data import iterator

import numpy as np

from loguru import logger


logger.add("logs.txt")
lines = open("logs.txt", "r").read()
for name, dataset in iterator():
    for metric in get_all_estimators():
        estimator = metric(dataset.x_train, dataset.y_train)
        print(f"Processing {name} - {estimator.__name__}")
        if f"'dataset': '{name}', 'metric': '{estimator.__name__}'" in lines:
            print(f"Skipping {name} - {estimator.__name__}")
            continue

        dataset.x_train = np.array(dataset.x_train)
        dataset.y_train = np.array(dataset.y_train)
        complexity = estimator.get_complexity()

        # Log as JSONL
        logger.info(
            {
                "dataset": name,
                "metric": estimator.__name__,
                "complexity": complexity,
            }
        )

        gc.collect()
