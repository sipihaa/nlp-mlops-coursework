import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

output_dir = "triton_repo/bert/1"
os.makedirs(output_dir, exist_ok=True)

# 1. Загружаем модель
model = joblib.load("models/classifier_model.pkl")

# 2. Определяем вход (размер 312 под BERT)
initial_type = [('float_input', FloatTensorType([None, 312]))]

# 3. Конвертируем с опцией zipmap=False !
# Это критически важно для Triton
options = {type(model): {'zipmap': False}}

onnx_model = convert_sklearn(
    model, 
    initial_types=initial_type, 
    options=options, # <--- ДОБАВИЛИ ВОТ ЭТО
    target_opset=15  # Оставляем 15, раз Triton 23.10 его понимает
)

# 4. Сохраняем
with open(f"{output_dir}/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Модель классификатора успешно конвертирована в ONNX")
