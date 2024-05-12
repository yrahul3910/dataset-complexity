import random

from metrics import BaseMetric
from metrics.types import ArrayLike, Float, NDArray, NpFloat, Preprocessor

import numpy as np

from keras.layers import Dense
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from tqdm import tqdm


class SmoothnessMetric(BaseMetric):
    def __init__(self, x_train: NDArray[NpFloat], y_train: ArrayLike):
        super().__init__(x_train, y_train)
        self.y_train = np.array(self.y_train)

    def _determine_n_class(self) -> int:
        if len(self.y_train.shape) > 1:
            return self.y_train.shape[1]

        if len(set(self.y_train)) <= 2:
            return 2

        # Otherwise, one-hot encode the values, and return the shape
        lb = LabelBinarizer()
        self.y_train = lb.fit_transform(self.y_train)
        return self.y_train.shape[1]

    def _get_random_model(self, n_class: int) -> Model:
        n_units = random.randint(3, 20)
        n_layers = random.randint(1, 5)

        model = Sequential()
        for _ in range(n_layers):
            model.add(Dense(n_units, activation="relu"))

        model.add(Dense(n_class, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def _get_random_preprocessor(self) -> Preprocessor:
        return random.choice([MinMaxScaler(), StandardScaler(), RobustScaler(), Normalizer()])

    def _max(self, betas: list) -> Float:
        # Return the maximum element that is not zero
        return max(list(filter(lambda x: x > 0, betas)))

    def get_complexity(self) -> Float:
        n_class = self._determine_n_class()
        BATCH_SIZE = 128
        N_TRIES = 30

        betas = []

        for _ in tqdm(range(N_TRIES)):
            model = self._get_random_model(n_class)
            preprocessor = self._get_random_preprocessor()
            self.x_train = preprocessor.fit_transform(self.x_train)

            model.fit(self.x_train, self.y_train, epochs=1, verbose=1, batch_size=BATCH_SIZE)

            def activ_func(xb):
                return Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)(xb)

            Ka_Kw = np.inf
            for i in range((len(self.x_train) - 1) // BATCH_SIZE + 1):
                start_i = i * BATCH_SIZE
                end_i = start_i + BATCH_SIZE
                xb = self.x_train[start_i:end_i]

                Ka = np.linalg.norm(activ_func([xb]))
                Kw = np.linalg.norm(model.layers[-1].weights[0])

                if not np.isinf(Ka / Kw):
                    Ka_Kw = min(Ka_Kw, Ka / Kw)

            betas.append((n_class - 1) / (n_class * BATCH_SIZE) * Ka_Kw)

        return self._max(betas)
