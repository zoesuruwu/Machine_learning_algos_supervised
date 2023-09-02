import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class Logistic_model:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        logger,
    ):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._logger = logger

    def fit_model(self):
        log_model = LogisticRegression()
        log_model = log_model.fit(self._X_train, self._y_train)
        return log_model

    def predict_test(self, log_model):
        y_pred = log_model.predict(self._X_test)
        print(classification_report(self._y_test, y_pred))
