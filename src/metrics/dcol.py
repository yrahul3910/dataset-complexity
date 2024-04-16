import shutil
import subprocess

from functools import partial

from metrics.base import BaseMetric
from metrics.types import ArrayLike, Float, NDArray, NpFloat

import arff
import numpy as np
import pandas as pd


def check_if_dcol_available():
    if shutil.which("dcol") is None:
        raise ValueError("Can`t find DCoL util, please add binary to $PATH."
                         "Install at https://github.com/nmacia/dcol")


class DCoL(BaseMetric):
    """DCoL Wrapper, do not use directly, use aliases below."""

    def __init__(self, x_train: NDArray[NpFloat], y_train: ArrayLike, metric: str, variant: str):
        super().__init__(x_train, y_train)
        self.metric = metric
        self.variant = variant

    def _dataset_to_arff(self):
        """Convert X_train, y_train to a valid ARFF file.

        Notes:
            DCoL requires the last column to be of type "enum" which can't be done
            with the python library.


        Returns:
            Path to ARFF file.
        """
        df = pd.DataFrame(data=self.x_train)
        path = "/tmp/dataset.arff"
        acc = [x.tolist() + [int(y)] for x, y in zip(self.x_train, self.y_train)]
        arff.dump('/tmp/tmp.arff'
                  , acc
                  , relation='relation name'
                  , names=list(df.columns) + ['class'])
        n_class = len(np.unique(self.y_train))
        # Replace class type for DCoL.
        with open(path, 'w') as f:
            subprocess.call(
                ['sed',
                 r"s/class integer/class \{" + ','.join([str(i) for i in range(n_class)]) + r"\}/g",
                 "/tmp/tmp.arff"], stdout=f
            )
        return path

    def get_complexity(self) -> Float:
        """
        Get complexity for `self.metric` `self.variant` using DCoL.

        Notes:
            You can compile DCoL from Github https://github.com/nmacia/dcol

        Returns:
            Complexity of dataset for `self.metric` `self.variant`.
        """
        check_if_dcol_available()
        ds_path = self._dataset_to_arff()
        subprocess.call([
            'dcol',
            '-i',
            ds_path,
            '-o',
            '/tmp/dcol_output',
            f'-{self.metric}',
            self.variant,
            '-xml'
        ])
        # The format is weird, need to load the output as XML and find the right column.
        return pd.read_xml('/tmp/dcol_output.xml').iloc[0][f'{self.metric}{self.variant}']


# Fisher Discriminant
F1 = partial(DCoL, metric='F', variant='1')
F1v = partial(DCoL, metric='F', variant='1v')
F2 = partial(DCoL, metric='F', variant='2')
F3 = partial(DCoL, metric='F', variant='3')
F4 = partial(DCoL, metric='F', variant='4')

# Linearity
L1 = partial(DCoL, metric='L', variant='1')
L2 = partial(DCoL, metric='L', variant='2')
L3 = partial(DCoL, metric='L', variant='3')

# Neighbours
N1 = partial(DCoL, metric='N', variant='1')
N2 = partial(DCoL, metric='N', variant='2')
N3 = partial(DCoL, metric='N', variant='3')
N4 = partial(DCoL, metric='N', variant='4')

# Dimensionality
T1 = partial(DCoL, metric='T', variant='1')
T2 = partial(DCoL, metric='T', variant='2')
