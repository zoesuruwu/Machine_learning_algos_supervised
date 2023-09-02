import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report


class Neural_model:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        epochs: int,
        num_nodes: int,
        dropout_prob: int,  # prevent overfitting
        lr: int,
        batch_size: int,
        logger,
    ):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._epochs = epochs
        self._batch_size = batch_size
        self._logger = logger
        self._num_nodes = num_nodes
        self._dropout_prob = dropout_prob
        self._lr = lr

    def plot_history(self, history):
        # val is performing worse than training_set
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history.history["loss"], label="loss")
        ax1.plot(history.history["val_loss"], label="val_loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Binary crossentropy")
        ax1.grid(True)

        ax2.plot(history.history["accuracy"], label="accuracy")
        ax2.plot(history.history["val_accuracy"], label="val_accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True)
        plt.show

    def fit_model(self):
        nn_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self._num_nodes, activation="relu", input_shape=(10,)
                ),
                tf.keras.layers.Dropout(self._dropout_prob),
                tf.keras.layers.Dense(self._num_nodes, activation="relu"),
                tf.keras.layers.Dropout(self._dropout_prob),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(self._lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        history = nn_model.fit(
            self._X_train,
            self._y_train,
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_split=0.2,
            verbose=0,
        )

        return nn_model, history
