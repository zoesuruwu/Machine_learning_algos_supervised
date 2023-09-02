import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report


class Linear_regression:
    def __init__(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame,
        test: pd.DataFrame,
        x_regressor: str,
        logger,
    ):
        self._train = train
        self._valid = valid
        self._test = test
        self._x_regressor = x_regressor
        self._logger = logger

    def get_xy(self, dataframe, y_label, x_labels=None):
        dataframe = copy.deepcopy(
            dataframe
        )  # This is because the copied object is not a reference to the original object; it's a whole new object with the same values
        if x_labels is None:  # take all regressors except for target variable
            X = dataframe[[c for c in dataframe.columns if c != y_label]].values
        else:
            if len(x_labels) == 1:
                X = dataframe[x_labels[0]].values.reshape(-1, 1)
            else:
                X = dataframe[x_labels].values

        y = dataframe[y_label].values.reshape(-1, 1)
        data = np.hstack((X, y))

        return data, X, y

    def fit_reg(self):
        # split into tain, validation, test data
        if isinstance(self._x_regressor, str):
            regressors = [self._x_regressor]
        else:
            regressors = self._x_regressor

        _, X_train, y_train = self.get_xy(
            self._train, "bike_count", x_labels=regressors
        )
        _, X_val, y_val = self.get_xy(self._valid, "bike_count", x_labels=regressors)
        _, self._X_test, self._y_test = self.get_xy(
            self._test, "bike_count", x_labels=regressors
        )
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)
        print("Coefficients, last entry is intercept: ")
        print(reg_model.coef_, reg_model.intercept_)
        print(f"R squred: {reg_model.score(self._X_test, self._y_test)}")

        return reg_model, X_train, y_train

    def compile_layers_plot(self, model, kind: str, lr: int, epo: int):
        self._logger.info(
            f"Neural network with mean squared error - layer kind: {kind}"
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mean_squared_error",
        )
        if self._X_train.shape[1] > 1:
            history = model.fit(
                self._X_train,
                self._y_train,
                verbose=0,
                epochs=epo,
                validation_data=(self._X_val, self._y_val),
            )
        else:
            history = model.fit(
                self._X_train.reshape(-1),
                self._y_train,
                verbose=0,
                epochs=epo,
                validation_data=(self._X_val, self._y_val),
            )

        self.plot_NN_loss(history)

    def _get_normalizer(self, num_row: int):
        normalizer = tf.keras.layers.Normalization(
            input_shape=(num_row,), axis=None
        )  # input will be batches of 1-dimensional vectors.
        if num_row > 1:
            normalizer.adapt(self._X_train)
        else:
            normalizer.adapt(self._X_train.reshape(-1))  # 1-dimensional vectors.
        return normalizer

    def fit_nn(self, num_regress):
        if isinstance(self._x_regressor, str):
            regressors = [self._x_regressor]
        else:
            regressors = self._x_regressor
        _, self._X_train, self._y_train = self.get_xy(
            self._train, "bike_count", x_labels=regressors
        )
        _, self._X_val, self._y_val = self.get_xy(
            self._valid, "bike_count", x_labels=regressors
        )
        _, X_test, y_test = self.get_xy(self._test, "bike_count", x_labels=regressors)

        normalizer_nn_model_linear_lay = tf.keras.Sequential(
            [
                self._get_normalizer(num_regress),
                tf.keras.layers.Dense(1),
            ]  # linear layer
        )

        normalizer_nn_model_multiple_lay = tf.keras.Sequential(
            [
                self._get_normalizer(num_regress),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        self.compile_layers_plot(
            normalizer_nn_model_linear_lay, kind="Single linear layer", lr=0.1, epo=1000
        )
        self.compile_layers_plot(
            normalizer_nn_model_multiple_lay,
            kind="Muliple layer with activation=relu",
            lr=0.001,
            epo=100,
        )

        return normalizer_nn_model_linear_lay, normalizer_nn_model_multiple_lay

    def compare_MSE(self, linear_reg, nn_model):
        self._logger.info("Comparing the MSE for both linear reg and nn")
        y_pred_lr = linear_reg.predict(self._X_test)
        y_pred_nn = nn_model.predict(self._X_test)
        MSE_lr = (np.square(y_pred_lr - self._y_test)).mean()
        MSE_nn = (np.square(y_pred_nn - self._y_test)).mean()
        print(f"Linear: {MSE_lr}, NN: {MSE_nn}")
        ax = plt.axes(aspect="equal")
        plt.scatter(self._y_test, y_pred_lr, label="Lin Reg Preds")
        plt.scatter(self._y_test, y_pred_nn, label="NN Preds")
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        lims = [0, 1800]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.legend()
        _ = plt.plot(lims, lims, c="red")

    def plot_prediction(self, X_train, y_train, reg):
        plt.scatter(X_train, y_train, label="Data", color="blue")
        x = tf.linspace(-20, 40, 100)
        plt.plot(
            x,
            reg.predict(np.array(x).reshape(-1, 1)),
            label="Fit",
            color="red",
            linewidth=3,
        )
        plt.legend()
        plt.title("Bikes vs Temp")
        plt.ylabel("Number of bikes")
        plt.xlabel("Temp")
        plt.show()

    def plot_NN_loss(self, history):
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        plt.show()
