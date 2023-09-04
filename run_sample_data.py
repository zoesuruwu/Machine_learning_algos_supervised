import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

from constants import (BATCH_SIZE, COL_NUM, COLS_BINARY, DROPOUT_OPTIONS,
                       EPOCHS, LR_OPTIONS, NUM_NODES_OPTIONS)
from helper_functions import plot_each_var, sample_data_standard
from knn import Knn_model
from linear_regression import Linear_regression
from log_regression import Logistic_model
from naive_bayes import Naive_bayes_model
from neural_net import Neural_model

sns.set_style("darkgrid")


def numeric_data(logger):
    df = pd.read_csv("data/BikeData.csv").drop(["Holiday", "Seasons"], axis=1)
    df["functional"] = (df["functional"] == "Yes").astype(int)
    # only look at noon
    df = df[df["hour"] == 12]
    df = df.drop(["hour"], axis=1)

    for label in df.columns[1:]:
        plt.figure(figsize=(10, 6), tight_layout=True)
        ax = sns.scatterplot(data=df, x=label, y="bike_count")
        ax.set(xlabel=label, ylabel="bike_count", title=label)
        plt.show()
    # after reading the plots
    df = df.drop(["wind", "visibility", "functional"], axis=1)
    train, valid, test = np.split(
        df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))]
    )
    logger.info("Starting with simple linear regression...")
    one_reg_model = Linear_regression(train, valid, test, "temp", logger)
    simple_l_r, X_train, y_train = one_reg_model.fit_reg()
    one_reg_model.plot_prediction(X_train, y_train, simple_l_r)
    logger.info("Starting with simple linear regression with Neural network...")
    nn_model_linear, nn_model_multi_relu = one_reg_model.fit_nn(num_regress=1)
    one_reg_model.plot_prediction(X_train, y_train, nn_model_linear)
    one_reg_model.plot_prediction(X_train, y_train, nn_model_multi_relu)

    logger.info("Starting with multiple linear regression...")
    multiple_model = Linear_regression(train, valid, test, df.columns[1:], logger)
    multi_l_r, X_train, y_train = multiple_model.fit_reg()
    logger.info("Starting with multiple linear regression with Neural network...")
    nn_model_linear, nn_model_multi_relu = multiple_model.fit_nn(
        num_regress=X_train.shape[1]
    )
    multiple_model.compare_MSE(multi_l_r, nn_model_multi_relu)


def binary_class(logger):
    df = pd.read_csv("data/magic04.data", names=COLS_BINARY)
    logger.info("Get info and statistics of data")
    print(df.info())
    print(df.describe())
    # chang the target variable into integer (0, 1)
    df["class"] = (df["class"] == "g").astype(int)
    plot_each_var(df)
    train, valid, test = np.split(
        df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))]
    )
    train, X_train, y_train = sample_data_standard(
        train, logger, "train", oversample=True
    )
    valid, X_valid, y_valid = sample_data_standard(
        valid, logger, "validation", oversample=False
    )
    test, X_test, y_test = sample_data_standard(test, logger, "test", oversample=False)

    logger.info("Starting with Knn model...")
    model = Knn_model(X_train, y_train, X_test, y_test, 1, logger)
    knn_model = model.fit_model()
    model.predict_test(knn_model)
    model = Knn_model(X_train, y_train, X_test, y_test, 3, logger)
    knn_model = model.fit_model()
    model.predict_test(knn_model)

    logger.info("Starting with Naive Bayes model...")
    model = Naive_bayes_model(X_train, y_train, X_test, y_test, logger)
    nb_model = model.fit_model()
    model.predict_test(nb_model)

    logger.info("Starting with logistic regression model...")
    model = Logistic_model(X_train, y_train, X_test, y_test, logger)
    log_model = model.fit_model()
    model.predict_test(log_model)

    logger.info("Starting with svm model...")
    model = Logistic_model(X_train, y_train, X_test, y_test, logger)
    log_model = model.fit_model()
    model.predict_test(log_model)

    # grid search
    logger.info("Starting with Neural model...")
    least_val_loss = float("inf")
    least_loss_model = None
    best_param = {
        "num_nodes": np.nan,
        "dropout_prob": np.nan,
        "lr": np.nan,
        "batch_size": np.nan,
    }
    for num_nodes in NUM_NODES_OPTIONS:
        for dropout_prob in DROPOUT_OPTIONS:
            for lr in LR_OPTIONS:
                for batch_size in BATCH_SIZE:
                    logger.info(
                        f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}"
                    )
                    nn_model = Neural_model(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        EPOCHS,
                        num_nodes,
                        dropout_prob,
                        lr,
                        batch_size,
                        logger,
                    )
                    fitted_nn_model, history = nn_model.fit_model()
                    nn_model.plot_history(history)
                    val_loss = fitted_nn_model.evaluate(X_valid, y_valid)[0]
                    if val_loss < least_val_loss:
                        least_val_loss = val_loss
                        least_loss_model = fitted_nn_model
                        best_param["num_nodes"] = num_nodes
                        best_param["dropout_prob"] = dropout_prob
                        best_param["lr"] = lr
                        best_param["batch_size"] = batch_size

    y_pred = least_loss_model.predict(X_test)
    y_pred = (
        (y_pred > 0.5)
        .astype(int)
        .reshape(
            -1,
        )
    )
    print(classification_report(y_test, y_pred))
    print(
        f"Best predictive NN model has the following parameter: \n{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}"
    )
