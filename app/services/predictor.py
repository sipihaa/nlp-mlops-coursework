import sys
import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data.preprocessor import preprocess_text

sys.path.append(os.getcwd()) 


class MLService:
    def __init__(self):
        self.bert_model = None
        self.classifier = None
        self.labels_map = {0: "Авиация", 1: "Автомобильный транспорт"}
        self._load_models()

    def _load_models(self):
        print("Loading models...")
        self.bert_model = SentenceTransformer('cointegrated/rubert-tiny2')
        
        model_path = "models/logreg_model.pkl"
        self.classifier = joblib.load(model_path)
        print("Модель загружена!")

    def predict(self, text: str) -> dict:
        clean_text = preprocess_text(text)
        if not clean_text:
            return {"label": "Unknown", "class_id": -1, "confidence": 0.0}

        embedding = self.bert_model.encode([clean_text])

        pred_id = self.classifier.predict(embedding)[0]
        
        try:
            probs = self.classifier.predict_proba(embedding)[0]
            confidence = float(np.max(probs))
        except AttributeError:
            confidence = 1.0

        return {
            "label": self.labels_map.get(pred_id, "Unknown"),
            "class_id": int(pred_id),
            "confidence": confidence
        }


ml_service = MLService()
