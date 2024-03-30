from metrics import BaseMetric
from metrics.types import ArrayLike, Float, NDArray, NpFloat

from spectral_metric.estimator import CumulativeGradientEstimator


class CSGMetric(BaseMetric):
    def __init__(self, x_train: NDArray[NpFloat], y_train: ArrayLike):
        super().__init__(x_train, y_train)
        self.estimator = CumulativeGradientEstimator(M_sample=250, k_nearest=5)

    def get_complexity(self) -> Float:
        self.estimator.fit(data=self.x_train, target=self.y_train)
        return self.estimator.csg
