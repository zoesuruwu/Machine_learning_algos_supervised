import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

from constants import COLS_BINARY


def plot_each_var(df):
    # Set Seaborn style
    sns.set()
    for each_col in COLS_BINARY[:-1]:
        plt.figure(figsize=(8, 6))
        plt.hist(
            df[df["class"] == 1][each_col],
            color="blue",
            label="gamma",
            alpha=0.7,
            density=True,
            bins=10,
        )
        plt.hist(
            df[df["class"] == 0][each_col],
            color="red",
            label="hadron",
            alpha=0.7,
            density=True,
            bins=10,
        )
        plt.title(each_col)
        plt.xlabel(each_col)
        plt.ylabel("Probability")
        plt.legend()
        plt.show()


def sample_data_standard(
    df: pd.DataFrame, logger, type_df: str, oversample: bool = False
):
    # 60 % train, 20% validation, 20% test data
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    class_1_before = sum(y == 1)
    class_0_before = sum(y == 0)
    # standardizing
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # resampleing
    if (
        oversample
    ):  # take the less class and resample, to increase the data size to make even
        logger.info(f"Resample {type_df} dataset...")
        logger.info(
            f"----- Before sampling, # observations having class = 1: {class_1_before}, # observations having class = 0: {class_0_before}"
        )
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
        class_1_after = sum(y == 1)
        class_0_after = sum(y == 0)
        logger.info(
            f"----- After sampling, # observations having class = 1: {class_1_after}, # observations having class = 0: {class_0_after}"
        )

    data = np.hstack((X, np.reshape(y, (-1, 1))))  # put side by side
    # data = np.hstack((X, np.reshape(y, (len(y), 1))))  # alternative
    return data, X, y
