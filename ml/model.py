from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from joblib import dump
from ml.data import process_data
import pandas as pd
import numpy as np


# Optional: implement hyperparameter tuning.
def train_model(train_data, categorical_features):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    X_train, y_train, encoder, lb = process_data(
        train_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # Define the model
    model = LogisticRegression()

    # Define the hyperparameters to tune
    hyperparameters = {'C': [0.1, 1, 10, 100],
                       'penalty': ['l2']}

    # Use a cross-validation grid search to tune the hyperparameters

    grid_search = GridSearchCV(model, hyperparameters, cv=5)
    grid_search.fit(X_train, y_train)

    dump(model, f"model/model.joblib")
    dump(encoder, f"model/encoder.joblib")
    dump(lb, f"model/lb.joblib")

    # Return the best performing model from the grid search
    return grid_search.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.linear_model
        Trained machine learning model.
    X : np.array
        "Data" used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def model_performance_on_categorical_slices(
        model, values: pd.DataFrame, target: np.ndarray):
    """
    Outputs the performance of a model on slices of a dataset where each slice contains only one unique value of a
    categorical feature.

    This function takes a model, a DataFrame of input values, and an array of target values as input, and it outputs
    the performance of the model on slices of the input values and target values where each slice contains only one
    unique value of a categorical feature.

    Parameters
    ----------
    model : object
        The model to evaluate.
    values : pd.DataFrame
        A DataFrame containing the input values. The DataFrame must have at least one categorical column.
    target : np.ndarray
        An array containing the target values.

    Returns
    -------
    None

    Example
    -------
    model = LogisticRegression()
    values = pd.DataFrame({'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                           'year': [2000, 2001, 2002, 2001, 2002],
                           'pop': [1.5, 1.7, 3.6, 2.4, 2.9]})
    target = np.array([0, 0, 1, 1, 1])
    model_performance_on_categorical_slices(model, values, target)
    """

    # Get the categorical features from the data
    categorical_features = [
        col for col in values.columns if values[col].dtype == "object"]

    # Loop through each categorical feature and output the performance of the model on slices of the data where each slice
    # contains only one unique value of the categorical feature
    for feature in categorical_features:
        # Get the unique values of the categorical feature
        unique_values = values[feature].unique()

        # Loop through each unique value and output the performance of the model on the data where the value of the
        # categorical feature is equal to the unique value
        for value in unique_values:
            mask = values[feature].isin([value])

            # Create a slice of the data where the value of the categorical
            # feature is equal to the unique value
            slice_data = values[mask]
            slice_target = target[mask]
            preds = inference(model, slice_data)
            # Output the performance of the model on the slice of the data
            print(
                f"Performance on {feature} = {value}: {compute_model_metrics(slice_target, preds)}")
