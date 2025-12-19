import os
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data.preprocessor import preprocess_text
import tritonclient.http as httpclient


class MLService:
    def __init__(self):
        self.bert_model = None
        self.labels_map = {0: "Авиация", 1: "Автомобильный транспорт"}
        self._load_model()

    def _load_model(self):
        print("Загрузка модели BERT...")

        model_path = "/app/rubert-tiny2" 
        if os.path.exists(model_path):
            self.bert_model = SentenceTransformer(model_path)
        else:
            self.bert_model = SentenceTransformer('cointegrated/rubert-tiny2')
            
        print("Модель BERT загружена!")

    def predict(self, text: str) -> dict:
        clean_text = preprocess_text(text)
        if not clean_text:
            return {"label": "Unknown", "class_id": -1, "confidence": 0.0}

        embedding = self.bert_model.encode([clean_text])

        triton_url = os.getenv("TRITON_URL", "localhost:8000")

        client = httpclient.InferenceServerClient(url=triton_url)

        inputs = [
            httpclient.InferInput("float_input", embedding.shape, "FP32")
        ]
        outputs = [
            httpclient.InferRequestedOutput("label"),
            httpclient.InferRequestedOutput("probabilities")
        ]
        inputs[0].set_data_from_numpy(embedding.astype(np.float32))

        results = client.infer(model_name="classifier", inputs=inputs, outputs=outputs)

        pred_id = int(results.as_numpy("label")[0])
        probs = results.as_numpy("probabilities")[0]

        return {
            "label": self.labels_map.get(pred_id, "Unknown"),
            "class_id": int(pred_id),
            "confidence": float(probs[pred_id])
        }


ml_service = MLService()
