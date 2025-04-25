# lab8_flask/app.py

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

app = FastAPI(title="WineRF Scoring")

# ─── 0) Load feature order you serialized at training time ───
# joblib.load returns (sklearn_model, feature_names_list)
_, FEATURE_NAMES = joblib.load("rf_model.joblib")

# ─── 1) Load the latest WineRF model from the MLflow registry ───
client = MlflowClient()
try:
    versions = client.search_model_versions("name='WineRF'")
    if not versions:
        raise RuntimeError("No registered versions for model 'WineRF'.")
    latest = max(versions, key=lambda mv: int(mv.version))
    model_uri = latest.source
    model = mlflow.pyfunc.load_model(model_uri)
except MlflowException as me:
    raise RuntimeError(f"MLflow registry error: {me}")
except Exception as e:
    raise RuntimeError(f"Could not load WineRF: {e}")

# ─── 2) Define your input schema with all 13 wine features ───
class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.post("/predict")
def predict(payload: WineInput):
    # 1) Turn incoming JSON into DataFrame
    df = pd.DataFrame([payload.dict()])

    # 2) If any original column had a slash ("/"), rename back:
    rename_map = {safe: orig
                  for orig in FEATURE_NAMES
                  if "/" in orig
                  for safe in [orig.replace("/", "_")]}
    if rename_map:
        df = df.rename(columns=rename_map)

    # 3) Select & reorder exactly as during training
    df = df[FEATURE_NAMES]

    # 4) Predict
    try:
        pred = model.predict(df)[0]
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))

    return {"prediction": int(pred)}
