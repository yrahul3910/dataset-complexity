from typing import TypeAlias

import numpy as np

from numpy import typing as npt
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler


Float: TypeAlias = np.float64 | float
NpFloat: TypeAlias = np.float64
NDArray: TypeAlias = npt.NDArray
ArrayLike: TypeAlias = npt.ArrayLike
Preprocessor: TypeAlias = MinMaxScaler | StandardScaler | RobustScaler | Normalizer
