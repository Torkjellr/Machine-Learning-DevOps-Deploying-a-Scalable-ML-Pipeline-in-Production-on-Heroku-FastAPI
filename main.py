# Put the code for your API here.
import numpy as np
import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from ml.model import inference

app = FastAPI()


# Define the Pydantic model for the POST request
class InferenceInput(BaseModel):
    data: np.array

# Create the FastAPI app
app = FastAPI()

# Define the GET request handler
@app.get('/')
async def welcome():
    return {"message": 'Welcome to the machine learning model API!'}

# Define the POST request handler
@app.post('/inference')
async def inference(input: InferenceInput):
    # load model
    model = joblib.load("model/model.joblib")

    # Run model inferences on the input data
    preds = inference(model, input.data)

    # Return the predictions
    return {'predictions': preds}
