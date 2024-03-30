from abc import ABC, abstractmethod

from metrics.types import ArrayLike, Float, NDArray, NpFloat


class BaseMetric(ABC):
    def __init__(self, x_train: NDArray[NpFloat], y_train: ArrayLike):
        self.x_train = x_train
        self.y_train = y_train

    @abstractmethod
    def get_complexity(self) -> Float:
        raise NotImplementedError
