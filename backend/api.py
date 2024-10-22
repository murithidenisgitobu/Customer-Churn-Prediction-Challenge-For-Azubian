from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
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
...
### Output features
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
    REGION: Optional[str] = Field(default=None)
    MRG: Optional[str] = Field(default=None)
    TOP_PACK: Optional[str] = Field(default=None)
    TENURE: Optional[str] = Field(default=None)
    MONTANT: Optional[float] = Field(default=None)
    FREQUENCE_RECH: Optional[float] = Field(default=None)
    REVENUE: Optional[float] = Field(default=None)
    ARPU_SEGMENT: Optional[float] = Field(default=None)
    FREQUENCE: Optional[float] = Field(default=None)
    DATA_VOLUME: Optional[float] = Field(default=None)
    ON_NET: Optional[float] = Field(default=None)
    ORANGE: Optional[float] = Field(default=None)
    TIGO: Optional[float] = Field(default=None)
    ZONE1: Optional[float] = Field(default=None)
    ZONE2: Optional[float] = Field(default=None)
    REGULARITY: Optional[int] = Field(default=None)
    FREQ_TOP_PACK: Optional[float] = Field(default=None)

@app.post("/predict")
async def predict_churn(model: Literal['xgboost', 'logistic_regression'], features: CustomerData):
    logger.info(f"Received prediction request for model: {model}")
    logger.debug(f"Input features: {features}")
    try:
        # Convert input features to a pandas DataFrame
        input_data = pd.DataFrame([features.dict(exclude_none=True)])  # Exclude None values
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
