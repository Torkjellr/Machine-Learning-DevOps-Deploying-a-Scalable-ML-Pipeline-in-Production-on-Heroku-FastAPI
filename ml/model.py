from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
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

    # Define the model
    model = LogisticRegression()

    # Define the hyperparameters to tune
    hyperparameters = {'C': [0.1, 1, 10, 100],
                       'penalty': ['l1', 'l2']}

    # Use a cross-validation grid search to tune the hyperparameters

    grid_search = GridSearchCV(model, hyperparameters, cv=5)
    grid_search.fit(X_train, y_train)

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
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def model_performance_on_categorical_slices(model, data):
    # Get the categorical features from the data
    categorical_features = [col for col in data.columns if data[col].dtype == "object"]

    # Loop through each categorical feature and output the performance of the model on slices of the data where each slice
    # contains only one unique value of the categorical feature
    for feature in categorical_features:
        # Get the unique values of the categorical feature
        unique_values = data[feature].unique()
        
        # Loop through each unique value and output the performance of the model on the data where the value of the
        # categorical feature is equal to the unique value
        for value in unique_values:
            # Create a slice of the data where the value of the categorical feature is equal to the unique value
            slice_data = data[data[feature] == value]
            
            # Output the performance of the model on the slice of the data
            print(f"Performance on {feature} = {value}: {inference(model, slice_data)}")