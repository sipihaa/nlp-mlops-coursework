from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictionRequest, PredictionResponse
from app.services.predictor import ml_service


router = APIRouter()

@router.get("/health")
def health_check():
    is_healthy = ml_service.check_health()
    
    if is_healthy:
        return {"status": "ok"}
    else:
        raise HTTPException(status_code=503, detail="Service Unavailable")

@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    try:
        result = ml_service.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
