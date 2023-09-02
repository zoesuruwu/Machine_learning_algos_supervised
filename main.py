import logging

from run_sample_data import binary_class, numeric_data

# Configure the logging settings
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    logger = logging.getLogger()
    logger.info("Numerical data - using models: Linear regression")
    numeric_data(logger)
    logger.info(
        "Binary class data - using models: Knn, Naive Bayes, logistic regression, svm, Neural"
    )
    binary_class(logger)


if __name__ == "__main__":
    main()
