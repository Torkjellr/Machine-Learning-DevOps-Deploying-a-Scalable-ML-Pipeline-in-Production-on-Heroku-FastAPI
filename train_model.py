import pandas as pd
# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.model import *
from ml.data import process_data


def train_test_model():
    """
    Trains a model on a training dataset, saves the trained model,
        and outputs the performance of the model on a test dataset.

    This function loads a dataset from a CSV file, splits the data
        into a training set and a test set, trains a model on
        the training set, saves the trained model to a file,
        and outputs the performance of the model on the test set.

    Returns
    -------
    None

    Example
    -------
    train_test_model()
    """

    # Add code to load in the data.
    data = pd.read_csv(f"data/census.csv")

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Train and save a model.
    model = train_model(train, categorical_features=cat_features)

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=True
    )

    model_performance_on_categorical_slices(model,
                                            inference(model, X_test),
                                            y_test
                                            )


if __name__ == "__main__":
    train_test_model()
