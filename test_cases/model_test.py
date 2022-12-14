from ml.model import train_model
from joblib import load
from sklearn.model_selection import train_test_split
from ml.data import process_data

def test_train_model_save(raw_data, cat_features):
    # Train and save a model.
    model = train_model(raw_data, cat_features)

    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    # Check that the model, encoder, and label binarizer have been saved correctly
    assert model is not None
    assert encoder is not None
    assert lb is not None

def test_process_data(raw_data, cat_features):
    X_train, y_train, encoder, lb = process_data(
        raw_data, 
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0