from fastapi import FastAPI
import pandas as pd

from api.schemas import LoanApplication
from src.models.predict import predict

app = FastAPI(
    title="Loan Risk API",
    version="1.0"
)


@app.get("/")
def home():

    return {
        "message": "Loan Risk API is running."
    }


@app.post("/predict")
def predict_loan(application: LoanApplication):

    threshold = application.threshold

    data = application.model_dump()

    data.pop("threshold")

    input_df = pd.DataFrame([data])

    result = predict(input_df, threshold)

    return result