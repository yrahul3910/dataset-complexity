from functools import partial
from itertools import product

from metrics.base import BaseMetric
from metrics.csg import CSGMetric
from metrics.dcol import F1, F2, F3, F4, L1, L2, L3, N1, N2, N3, N4, T1, T2, F1v
from metrics.smoothness import SmoothnessMetric
from metrics.strong_convexity import StrongConvexityMetric
from metrics.types import ArrayLike, Float, NDArray, NpFloat


all_estimators = [
    ("CSGMetric", CSGMetric),
    ("F1", F1),
    ("F2", F2),
    ("F3", F3),
    ("F4", F4),
    ("L1", L1),
    ("L2", L2),
    ("L3", L3),
    ("N1", N1),
    ("N2", N2),
    ("N3", N3),
    ("N4", N4),
    ("T1", T1),
    ("T2", T2),
    ("F1v", F1v),
    ("SmoothnessMetric", SmoothnessMetric),
    ("StrongConvexityMetric", StrongConvexityMetric),
]

"""
Stores kwargs for estimators, in case any need them. For now, none of them do.
Each estimator name maps to a list of kwarg-possible value maps, e.g.
{
   "CSGMetric": {
       "M_sample": [100, 250],
       "k_nearest": [3, 5]
   }
}
"""
estimator_kwargs = {name: {} for name, _ in all_estimators}


def get_all_kwargs_for_estimator(estimator_name: str):
    keys = estimator_kwargs[estimator_name].keys()
    for values in product(*estimator_kwargs[estimator_name].values()):
        yield dict(zip(keys, values))


def get_all_estimators():
    for name, estimator in all_estimators:
        # Yield every possible combo of estimator and kwargs
        for kwargs in get_all_kwargs_for_estimator(name):
            yield partial(estimator, **kwargs)


__all__ = [
    "ArrayLike",
    "BaseMetric",
    "CSGMetric",
    "Float",
    "NDArray",
    "NpFloat",
    "SmoothnessMetric",
    "StrongConvexityMetric",
    "F1",
    "F1v",
    "F2",
    "F3",
    "F4",
    "L1",
    "L2",
    "L3",
    "N1",
    "N2",
    "N3",
    "N4",
    "T1",
    "T2",
]
