import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import yaml
from dotenv import load_dotenv


load_dotenv()

mlflow_tracking_uri = "file:./mlruns"
artifact_root = "s3://nlp-mlops-coursework/mlflow-artifacts"

mlflow.set_tracking_uri(mlflow_tracking_uri)

experiment_name = "vk_classification"
run_name = "clean-logreg-v3"
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
    data = pd.read_pickle("data/processed/data_with_embeddings.pkl")

    with open("configs/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_params = config["model_params"]
    split_params = config["split_params"]

    X = list(data['embeddings'])
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=split_params["test_size"], 
        random_state=split_params["random_state"]
    )

    print("Начало обучения...")
    
    with mlflow.start_run(run_name=run_name):        
        mlflow.log_params(model_params)
        mlflow.log_params(split_params)

        clf = LogisticRegression(**model_params) 
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=clf,
            name="classifier_model",
            registered_model_name="classifier_model"
        )
        
        print(f"Модель сохранена в S3: {artifact_root}/{mlflow.active_run().info.run_id}")
    
    joblib.dump(clf, 'models/classifier_model.pkl')

    print("Модель сохранена локально")


if __name__ == "__main__":
    train()