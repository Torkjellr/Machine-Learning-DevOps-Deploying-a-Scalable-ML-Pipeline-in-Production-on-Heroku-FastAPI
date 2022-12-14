import pytest
import pandas as pd

@pytest.fixture
def raw_data():
    # Add code to load in the data.
    return pd.read_csv(f"starter/data/census.csv")

@pytest.fixture
def clean_data(raw_data):
    X_train, y_train, encoder, lb = process_data(
        train_data, 
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]