from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and scaler
model = joblib.load("loan_default_rf_model.pkl")
scaler = joblib.load("loan_default_scaler.pkl")

# Input schema
class LoanInput(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: int
    EmploymentType: int
    MaritalStatus: int
    HasMortgage: int
    HasDependents: int
    LoanPurpose: int
    HasCoSigner: int

@app.get("/")
def root():
    return {"message": "Loan Default Prediction API is running."}

@app.post("/predict")
def predict_default(data: LoanInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        scaled_input = scaler.transform(input_df)

        # Predict probability of default (class 1)
        prob_default = model.predict_proba(scaled_input)[0][1]

        # Custom threshold
        threshold = 0.3
        prediction = int(prob_default > threshold)

        return {
            "default_prediction": prediction,
            "probability": round(prob_default, 3),
            "threshold_used": threshold,
            "status": "success"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
