"""
Online inference of the model described in the first homework
"""
import os

import numpy as np

import uvicorn
from fastapi import FastAPI

from src.entities.inference_parameters import Request, Response
from src.models.fit_predict_model import predict_model
from src.models.model_dump import load_model

PATH_TO_MODEL: str = "models/model.pkl"
app = FastAPI()


@app.get("/")
def main():
    """
    The root get
    :return:
    """
    return {"What's happening ": "The app is starting"}


@app.get("/predict/", response_model=Response)
def predict(request: Request):
    """
    The functional get
    :param request:
    :return:
    """
    model_path = PATH_TO_MODEL
    model = load_model(model_path)
    return {"target": predict_model(model, np.array(request.data).reshape(1, -1))[0]}


if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=os.getenv("PORT", 8000)
    )
