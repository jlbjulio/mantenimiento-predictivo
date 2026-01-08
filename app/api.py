from fastapi import FastAPI
from pydantic import BaseModel
import os, sys, joblib, pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.data_loader import engineer_features, load_dataset

MODELS_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'failure_binary_model.joblib')
MULTI_PATH = os.path.join(MODELS_DIR, 'failure_multilabel_models.joblib')

app = FastAPI(title="API Mantenimiento Predictivo", version="1.1")
model = joblib.load(MODEL_PATH)
# Modelos multilabel se exponen solo vía endpoint /predict_modes; la UI Streamlit no los usa
multi_models = joblib.load(MULTI_PATH) if os.path.exists(MULTI_PATH) else None

class PredictRequest(BaseModel):
    air_temp_k: float
    process_temp_k: float
    rot_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    type: str

@app.get('/health')
def health():
    return {"status":"ok"}

@app.post('/predict')
def predict(req: PredictRequest):
    data = req.dict()
    full_df = load_dataset()
    # Exclude technical fields that shouldn't affect prediction
    # Keep all non-target features to maintain compatibility with models trained with product_id/uid
    feature_cols = [c for c in full_df.columns if c not in ['machine_failure','twf','hdf','pwf','osf','rnf']]
    template = {c: None for c in feature_cols}
    template.update(data)
    # Technical fields that are not used by the model are kept for traceability, but excluded by preprocessor
    template['uid'] = 0
    template['product_id'] = f"{data['type']}_API"
    row = pd.DataFrame([template])
    row = engineer_features(row)
    for c in feature_cols:
        if c not in row.columns:
            row[c] = None
    row = row[feature_cols]
    prob = model.predict_proba(row)[0][1]
    pred = int(model.predict(row)[0])
    return {"prediction": pred, "probability": prob}

@app.post('/predict_modes')
def predict_modes(req: PredictRequest):
    if multi_models is None:
        return {"error": "Modelos multilabel no disponibles"}
    data = req.dict()
    full_df = load_dataset()
    # Preparar fila base
    feature_cols = [c for c in full_df.columns if c not in ['machine_failure','twf','hdf','pwf','osf','rnf']]
    template = {c: None for c in feature_cols}
    template.update(data)
    template['uid'] = 0
    template['product_id'] = f"{data['type']}_API"
    row = pd.DataFrame([template])
    row = engineer_features(row)
    for c in feature_cols:
        if c not in row.columns:
            row[c] = None
    row = row[feature_cols]
    probs = {}
    for label, mdl in multi_models.items():
        try:
            probs[label] = float(mdl.predict_proba(row)[0][1])
        except Exception:
            probs[label] = None
    # Probabilidad global (si binario disponible)
    try:
        global_prob = float(model.predict_proba(row)[0][1])
    except Exception:
        global_prob = None
    return {"global_probability": global_prob, "mode_probabilities": probs}
