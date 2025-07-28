from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model = joblib.load("loan_default_rf_model.pkl")
scaler = joblib.load("loan_default_scaler.pkl")

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

@app.post("/predict")
def predict_default(data: LoanInput):
    input_df = pd.DataFrame([data.dict()])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    return {"default_prediction": int(prediction[0])}