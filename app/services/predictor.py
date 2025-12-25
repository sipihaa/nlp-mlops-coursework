import os
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data.preprocessor import preprocess_text
import tritonclient.http as httpclient


class MLService:
    def __init__(self):
        self.bert_model = SentenceTransformer('cointegrated/rubert-tiny2')
        self.labels_map = {0: "Авиация", 1: "Автомобильный транспорт"}
        self.triton_url = os.getenv("TRITON_URL", "localhost:8000")


    def check_health(self) -> bool:
        try:
            client = httpclient.InferenceServerClient(url=self.triton_url)
            
            if not client.is_server_ready():
                print("Triton server is not ready")
                return False
            
            if not client.is_model_ready("classifier"):
                print("Model 'classifier' is not ready")
                return False
                
            return True

        except Exception as e:
            print(f"Health check failed: {e}")
            return False
            

    def predict(self, text: str) -> dict:
        clean_text = preprocess_text(text)
        if not clean_text:
            return {"label": "Unknown", "class_id": -1, "confidence": 0.0}

        embedding = self.bert_model.encode([clean_text])

        client = httpclient.InferenceServerClient(url=self.triton_url)

        inputs = [
            httpclient.InferInput("float_input", embedding.shape, "FP32")
        ]
        outputs = [
            httpclient.InferRequestedOutput("label"),
            httpclient.InferRequestedOutput("probabilities")
        ]
        inputs[0].set_data_from_numpy(embedding.astype(np.float32))

        results = client.infer(model_name="classifier", inputs=inputs, outputs=outputs)

        pred_id = results.as_numpy("label")[0].item()
        probs = results.as_numpy("probabilities")[0]

        return {
            "label": self.labels_map.get(pred_id, "Unknown"),
            "class_id": int(pred_id),
            "confidence": float(probs[pred_id])
        }


ml_service = MLService()
