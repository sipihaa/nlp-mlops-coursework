import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

output_dir = "triton_repo/classifier/1"
os.makedirs(output_dir, exist_ok=True)

model = joblib.load("models/classifier_model.pkl")

initial_type = [('float_input', FloatTensorType([None, 312]))]

options = {type(model): {'zipmap': False}}

onnx_model = convert_sklearn(
    model, 
    initial_types=initial_type, 
    options=options,
    target_opset=15
)

with open(f"{output_dir}/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Модель классификатора успешно конвертирована в ONNX")
