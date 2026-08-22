from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

app = FastAPI(
    title="Bank Marketing Subscription Prediction API",
    description=(
        "REST API for predicting whether a bank customer "
        "is likely to subscribe to a term deposit."
    ),
    version="1.0.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "bank_marketing_model.joblib"


# ---------------------------------------------------------
# Load trained ML pipeline
# ---------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model = None
    model_error = str(exc)


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------

class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Bank Marketing Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_unavailable"
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        data = customer.model_dump()

        # Compatibility with trained model feature names
        if "cons_conf_idx" in data:
            data["cons.conf.idx"] = data.pop("cons_conf_idx")

        if "nr_employed" in data:
            data["nr.employed"] = data.pop("nr_employed")

        if "emp_var_rate" in data:
            data["emp.var.rate"] = data.pop("emp_var_rate")

        if "cons_price_idx" in data:
            data["cons.price.idx"] = data.pop("cons_price_idx")

        if "euribor3m" in data:
            data["euribor3m"] = data["euribor3m"]

        # Feature engineered during model development
        data["previously_contacted"] = int(data["pdays"] != 999)

        # These features were categorical during training
        categorical_cols = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "pdays",
            "poutcome",
        ]

        for col in categorical_cols:
            data[col] = str(data[col])

        input_data = pd.DataFrame([data])

        prediction = int(model.predict(input_data)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_data)[0][1])

        return {
            "prediction": prediction,
            "subscription": "yes" if prediction == 1 else "no",
            "probability": probability,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        )