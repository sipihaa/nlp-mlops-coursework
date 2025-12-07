import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from dotenv import load_dotenv


# Загружаем секреты (ключи S3)
load_dotenv()

# --- КОНФИГУРАЦИЯ ---
# Указываем, куда MLflow будет складывать артефакты (модели)
# Замените 'nlp-mlops-coursework' на имя вашего бакета
# MLflow создаст там папку mlruns
mlflow_tracking_uri = "file:./mlruns" # Логи (метрики) храним локально
artifact_root = "s3://nlp-mlops-coursework/mlflow-artifacts" # Артефакты в S3

# Настраиваем MLflow
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Создаем эксперимент (если нет) с указанием S3 пути для артефактов
experiment_name = "vk_classification"
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_root
    )
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow.set_experiment(experiment_name)

def train():
    print("Загрузка данных...")
    # Загружаем подготовленный датасет (из preprocessor.py)
    # Предполагаем, что там лежат эмбеддинги
    data = pd.read_pickle("data/processed/data_with_embeddings.pkl")
    
    X = list(data['embeddings']) # Превращаем колонку в список массивов
    y = data['y']

    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Параметры модели (лучше вынести в configs/train_config.yaml)
    params = {
        "random_state": 42,
        "max_iter": 10000,
        "C": 1.0,           # Сила регуляризации
        "solver": "lbfgs"
    }

    print("Начало обучения...")
    
    # --- ЗАПУСК MLFLOW RUN ---
    with mlflow.start_run():
        # 1. Логируем параметры
        mlflow.log_params(params)

        # 2. Обучение
        clf = LogisticRegression(**params)
        clf.fit(X_train, y_train)

        # 3. Предсказание и метрики
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # 4. Логируем метрики
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 5. Логируем модель (Она улетит в S3!)
        # registered_model_name зарегистрирует её в Model Registry (опционально)
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="logreg_model",
            registered_model_name="VK_Classifier_LogReg"
        )
        
        print(f"Модель сохранена в S3: {artifact_root}/{mlflow.active_run().info.run_id}/artifacts/logreg_model")

if __name__ == "__main__":
    train()