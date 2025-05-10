from pydantic import BaseModel
from typing import List

class SymptomRequest(BaseModel):
    symptoms: List[str]

# class PredictionResponse(BaseModel):
#     disease: str
#     confidence: float

class DiseasePrediction(BaseModel):
    disease: str
    confidence: float

class PredictionResponse(BaseModel):
    top_predictions: List[DiseasePrediction]
