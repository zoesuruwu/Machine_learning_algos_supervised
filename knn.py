import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


class Knn_model:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        n_neighbor: int,
        logger,
    ):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._logger = logger
        self._n_neighbor = n_neighbor

    def fit_model(self):
        knn_model = KNeighborsClassifier(n_neighbors=self._n_neighbor)
        knn_model.fit(self._X_train, self._y_train)
        return knn_model

    def predict_test(self, knnmodel):
        y_pred = knnmodel.predict(self._X_test)
        self._logger.info(
            f"Classification report based on n_neighbor: {self._n_neighbor}: y_test vs. y_predict"
        )
        print(classification_report(self._y_test, y_pred))
