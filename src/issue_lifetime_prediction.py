import warnings

import numpy as np
import pandas as pd

from raise_utils.data import Data
from raise_utils.transforms import Transform
from sklearn.model_selection import train_test_split
from spectral_metric.estimator import CumulativeGradientEstimator


def split_data(filename: str, data: Data, n_classes: int) -> Data:
    if n_classes == 2:
        if filename == "firefox.csv":
            data.y_train = data.y_train < 4
            data.y_test = data.y_test < 4
        elif filename == "chromium.csv":
            data.y_train = data.y_train < 5
            data.y_test = data.y_test < 5
        else:
            data.y_train = data.y_train < 6
            data.y_test = data.y_test < 6
    elif n_classes == 3:
        data.y_train = np.where(data.y_train < 2, 0, np.where(data.y_train < 6, 1, 2))
        data.y_test = np.where(data.y_test < 2, 0, np.where(data.y_test < 6, 1, 2))
    elif n_classes == 5:
        data.y_train = np.where(
            data.y_train < 1,
            0,
            np.where(data.y_train < 3, 1, np.where(data.y_train < 6, 2, np.where(data.y_train < 21, 3, 4))),
        )
        data.y_test = np.where(
            data.y_test < 1,
            0,
            np.where(data.y_test < 3, 1, np.where(data.y_test < 6, 2, np.where(data.y_test < 21, 3, 4))),
        )
    elif n_classes == 7:
        data.y_train = np.where(
            data.y_train < 1,
            0,
            np.where(
                data.y_train < 2,
                1,
                np.where(
                    data.y_train < 3,
                    2,
                    np.where(data.y_train < 6, 3, np.where(data.y_train < 11, 4, np.where(data.y_train < 21, 5, 6))),
                ),
            ),
        )
        data.y_test = np.where(
            data.y_test < 1,
            0,
            np.where(
                data.y_test < 2,
                1,
                np.where(
                    data.y_test < 3,
                    2,
                    np.where(data.y_test < 6, 3, np.where(data.y_test < 11, 4, np.where(data.y_test < 21, 5, 6))),
                ),
            ),
        )
    else:
        data.y_train = np.where(
            data.y_train < 1,
            0,
            np.where(
                data.y_train < 2,
                1,
                np.where(
                    data.y_train < 3,
                    2,
                    np.where(
                        data.y_train < 4,
                        3,
                        np.where(
                            data.y_train < 6,
                            4,
                            np.where(
                                data.y_train < 8, 5, np.where(data.y_train < 11, 6, np.where(data.y_train < 21, 7, 8))
                            ),
                        ),
                    ),
                ),
            ),
        )
        data.y_test = np.where(
            data.y_test < 1,
            0,
            np.where(
                data.y_test < 2,
                1,
                np.where(
                    data.y_test < 3,
                    2,
                    np.where(
                        data.y_test < 4,
                        3,
                        np.where(
                            data.y_test < 6,
                            4,
                            np.where(
                                data.y_test < 8, 5, np.where(data.y_test < 11, 6, np.where(data.y_test < 21, 7, 8))
                            ),
                        ),
                    ),
                ),
            ),
        )

    transform = Transform("normalize")
    transform.apply(data)

    return data


def load_issue_lifetime_prediction_data(filename: str, n_classes: int) -> Data:
    df = pd.read_csv(f"../../data/issue_close_time_prediction/bugzilla/{filename}.csv")
    df.drop(["Unnamed: 0", "bugID"], axis=1, inplace=True)
    _df = df[["s1", "s2", "s3", "s4", "s5", "s6", "s8", "y"]]
    _df["s70"] = df["s7"].apply(lambda x: eval(x)[0])
    _df["s71"] = df["s7"].apply(lambda x: eval(x)[1])
    _df["s72"] = df["s7"].apply(lambda x: eval(x)[2])
    _df["s90"] = df["s9"].apply(lambda x: eval(x)[0])
    _df["s91"] = df["s9"].apply(lambda x: eval(x)[1])

    if filename == "firefox":
        _df["s92"] = df["s9"].apply(lambda x: eval(x)[2])

    x = _df.drop("y", axis=1)
    y = _df["y"]

    data = Data(*train_test_split(x, y))
    data = split_data(filename, data, n_classes)
    return data


warnings.filterwarnings("ignore")
for filename in ["chromium", "eclipse", "firefox"]:
    for n_class in [2, 3, 5, 7, 9]:
        dataset = load_issue_lifetime_prediction_data(filename, n_class)

        estimator = CumulativeGradientEstimator(M_sample=250, k_nearest=5)
        estimator.fit(data=dataset.x_train, target=dataset.y_train)

        print(f"{filename}-{n_class}class: {estimator.csg}")
