# Put the code for your API here.
import os
import joblib

from fastapi import FastAPI
from ml.model import inference
from schema import InferenceInput

app = FastAPI()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Create the FastAPI app
app = FastAPI()

# Define the GET request handler


@app.get('/')
async def welcome():
    """
    This function provides a welcome message for users accessing the API.

    Returns:
        dict: A dictionary containing a welcome message for users.
    """
    return {"message": 'Welcome to the machine learning model API!'}


# Define the POST request handler


@app.post('/infere')
async def infere(data: InferenceInput):
    """
    This function accepts data as input and returns the result of
    running inference on the data using a pre-defined machine learning model.

    Args:
        data (InferenceInput): An object containing the data to be passed to
        the model for inference.

    Returns:
        dict: A dictionary containing the result of running inference on the
        data using the machine learning model.
    """

    data = data.dict()

    # rename _ names to -

    pedantic_keys = ["education_num",
                     "marital_status",
                     "capital_gain",
                     "capital_loss",
                     "hours_per_week"
                     ]
    new_keys = [
        "education-num",
        "marital-status",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]
    for ped_key, new_key in zip(pedantic_keys, new_keys):
        data[new_key] = data.pop(ped_key)

    # load model
    model = joblib.load("model/model.joblib")

    # Run model inferences on the input data
    preds = inference(model, data.data)

    # Return the predictions
    return {'predictions': preds}
