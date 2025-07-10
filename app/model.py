import joblib
import pandas as pd

model = joblib.load("pole_model.joblib")

def predict_pole(input_row: dict):
    df = pd.DataFrame([input_row])
    pred_probs = model.predict_proba(df)[0]
    pred_driver_id = model.classes_[pred_probs.argmax()]
    confidence = pred_probs.max()
    return pred_driver_id, confidence
