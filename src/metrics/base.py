from abc import ABC, abstractmethod
from copy import deepcopy

from metrics.types import ArrayLike, Float, NDArray, NpFloat


class BaseMetric(ABC):
    def __init__(self, x_train: NDArray[NpFloat], y_train: ArrayLike):
        self.x_train = deepcopy(x_train)
        self.y_train = deepcopy(y_train)

    @abstractmethod
    def get_complexity(self) -> Float:
        raise NotImplementedError
