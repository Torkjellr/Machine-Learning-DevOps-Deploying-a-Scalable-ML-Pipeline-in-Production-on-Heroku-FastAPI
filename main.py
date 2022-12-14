# Put the code for your API here.
import numpy as np
import joblib
import os

from fastapi import FastAPI

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
    return {"message": 'Welcome to the machine learning model API!'}

# Define the POST request handler
@app.post('/infere')
async def infere(input: InferenceInput):

    data = input.dict()

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
    for k, nk in zip(pedantic_keys, new_keys):
        data[nk] = data.pop(k)


    # load model
    model = joblib.load("model/model.joblib")

    # Run model inferences on the input data
    preds = inference(model, input.data)

    # Return the predictions
    return {'predictions': preds}
