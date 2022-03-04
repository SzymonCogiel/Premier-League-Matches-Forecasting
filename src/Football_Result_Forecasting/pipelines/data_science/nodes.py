# Data science model nodes

import logging
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def split_data(df: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = df.drop(columns=["FTR", "FTHG", "FTAG", "Date"]).copy()
    y = df["FTR"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for match result.

    Returns:
        Trained model.
    """
    classifier = LogisticRegression(C=5, multi_class='multinomial', solver='lbfgs')
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_model(
    classifier: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
):
    """Calculates and logs the accuracy.

    Args:
        classifier: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for match result.
    """
    y_pred = classifier.predict(X_test)
    a = accuracy_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a accuracy of %.3f on test data.", a)

