from fastapi import APIRouter
from app.schemas import SymptomRequest, PredictionResponse
from app.ml_model import get_disease

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_disease(request: SymptomRequest):
    result = get_disease(request.symptoms)
    return result