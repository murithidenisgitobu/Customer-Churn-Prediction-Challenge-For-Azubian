from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import joblib
from fastapi.responses import PlainTextResponse
import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

description = """
### Input Features

| Variable         | Description                                       |
|------------------|---------------------------------------------------|
| REGION           | The location of each client                       |
| MRG              | A client who is going                             |
| TOP_PACK         | The most active packs                             |
| TENURE           | Duration in the network (as a string, e.g. '3-6 months') |
| MONTANT          | Top-up amount                                     |
| FREQUENCE_RECH   | Number of times the customer refilled             |
| REVENUE          | Monthly income of each client                     |
| ARPU_SEGMENT     | Income over 90 days / 3                           |
| FREQUENCE        | Number of times the client has made an income     |
| DATA_VOLUME      | Number of connections                             |
| ON_NET           | Inter expresso call                               |
| ORANGE           | Call to orange                                    |
| TIGO             | Call to Tigo                                      |
| ZONE1            | Call to zones1                                    |
| ZONE2            | Call to zones2                                    |
| REGULARITY       | Number of times the client is active for 90 days  |
| FREQ_TOP_PACK    | Number of times the client has activated the top pack packages |

### Output features
[0] Will not Churn
[1] Churner
"""

app = FastAPI(title="Churn Predictor Backend with FastAPI", description=description)

@app.get("/", response_class=PlainTextResponse)
def read_root():
    return "Welcome to the Churn Predictor API!\nAdd '/redoc' or '/docs' to view documentation"

# Load models
try:
    xgb_path = os.path.join('..', 'Models Directory', 'xgb_model.joblib')
    lr_path = os.path.join('..', 'Models Directory', 'lr_model.joblib')

    xgboost = joblib.load(xgb_path)
    logistic_regression = joblib.load(lr_path)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise HTTPException(status_code=500, detail="Error loading models")

class CustomerData(BaseModel):
    REGION: str
    MRG: str
    TOP_PACK: str
    TENURE: str
    MONTANT: float
    FREQUENCE_RECH: int
    REVENUE: float
    ARPU_SEGMENT: float
    FREQUENCE: int
    DATA_VOLUME: float
    ON_NET: int
    ORANGE: int
    TIGO: int
    ZONE1: int
    ZONE2: int
    REGULARITY: int
    FREQ_TOP_PACK: int

@app.post("/Predict/")
async def predict_churn(model: Literal['xgboost', 'logistic_regression'], features: CustomerData):
    logger.info(f"Received prediction request for model: {model}")
    logger.debug(f"Input features: {features}")

    try:
        # Convert input features to a pandas DataFrame
        input_data = pd.DataFrame([features.dict()])
        logger.debug(f"Processed input data: {input_data}")

        # Select model based on the user's choice
        if model == 'xgboost':
            prediction = xgboost.predict(input_data)[0]
            probabilities = xgboost.predict_proba(input_data)[0]
        elif model == 'logistic_regression':
            prediction = logistic_regression.predict(input_data)[0]
            probabilities = logistic_regression.predict_proba(input_data)[0]
        else:
            raise ValueError("Invalid model specified")

        # Get the probability of the predicted outcome
        probability = probabilities[prediction]

        logger.info(f"Prediction result: {prediction}")
        logger.info(f"Prediction probability: {probability}")

        return {
            "model": model,
            "prediction": "Churn" if prediction == 1 else "Stay",
            "probability": float(probability)
        }
    except ValueError as ve:
        logger.error(f"ValueError in predict_churn: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error in predict_churn: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")