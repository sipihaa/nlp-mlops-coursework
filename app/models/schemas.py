from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., example="Вчера летал на Боинге в Сочи")

class PredictionResponse(BaseModel):
    label: str
    class_id: int
    confidence: float
