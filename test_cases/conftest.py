import pytest
import pandas as pd


@pytest.fixture
def raw_data():
    # Add code to load in the data.
    return pd.read_csv(f"data/census.csv")


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
