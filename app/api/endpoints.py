from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.predictor import ml_service


router = APIRouter()

@router.get("/health")
def health_check():
    if ml_service.classifier is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ok"}

@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    try:
        result = ml_service.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
