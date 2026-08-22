import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Creditcard_Model_Tuning"

mlflow.set_tracking_uri(TRACKING_URI)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Эксперимент '{EXPERIMENT_NAME}' не найден")

runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_auprc DESC"]
)
if runs_df.empty:
    raise RuntimeError("В эксперименте нет запусков")

best_run = runs_df.iloc[0]
best_run_id = best_run["run_id"]
best_auprc = best_run["metrics.test_auprc"]

print(f"Загружаю модель из лучшего запуска: {best_run_id}")
print(f"Лучшая метрика test_auprc: {best_auprc:.4f}")

model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/best_model")

class InputData(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Hour: int

class Prediction(BaseModel):
    pred: int
    prob: float

@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    return {
        "model_type": type(model).__name__,
        "is_pipeline": True
    }

@app.post('/predict', response_model=Prediction)
def predict(form: InputData):
    df = pd.DataFrame.from_dict([form.dict()])
    y_proba = model.predict_proba(df)[0][1]
    y_pred = int(model.predict(df)[0])
    return {"pred": y_pred, "prob": float(y_proba)}