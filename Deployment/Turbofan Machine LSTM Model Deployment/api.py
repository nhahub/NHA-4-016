from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np

model = tf.keras.models.load_model("my_lstm_model_w128_b16.keras")
ops_scaler = joblib.load("ops_scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
sequence_length = joblib.load("sequence_length.pkl")
regime_scalers = joblib.load("regime_scalers.pkl")
sensor_cols = joblib.load("sensor_cols.pkl")

_kmeans_raw = joblib.load("kmeans.pkl")
kmeans_centers = _kmeans_raw.cluster_centers_.astype(np.float32)  # extract just the numpy array

app = FastAPI()

class PredictionInput(BaseModel):
    op1: float
    op2: float
    op3: float

    s2: float
    s3: float
    s4: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s20: float
    s21: float


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
async def predict(input: PredictionInput):
        df = pd.DataFrame([{
            "op_setting_1": input.op1,
            "op_setting_2": input.op2,
            "op_setting_3": input.op3,
            "sensor_2": input.s2,
            "sensor_3": input.s3,
            "sensor_4": input.s4,
            "sensor_6": input.s6,
            "sensor_7": input.s7,
            "sensor_8": input.s8,
            "sensor_9": input.s9,
            "sensor_10": input.s10,
            "sensor_11": input.s11,
            "sensor_12": input.s12,
            "sensor_13": input.s13,
            "sensor_14": input.s14,
            "sensor_15": input.s15,
            "sensor_16": input.s16,
            "sensor_17": input.s17,
            "sensor_20": input.s20,
            "sensor_21": input.s21
        }])

        df = df.astype(np.float32)
        # APPLY SAME PREPROCESSING


        ops_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']


        df[ops_cols] = ops_scaler.transform(df[ops_cols]).astype(np.float32)

        # add regime
        ops = np.ascontiguousarray(
        df[['op_setting_1', 'op_setting_2', 'op_setting_3']].values, 
        dtype=np.float32)
        # Manual KMeans prediction: find nearest centroid using Euclidean distance
        distances = np.linalg.norm(ops[:, np.newaxis, :] - kmeans_centers[np.newaxis, :, :], axis=2)
        df['regime'] = np.argmin(distances, axis=1)

        for regime_id in range(6):
            mask = df['regime'] == regime_id
            
            if not mask.any():
                continue

            df.loc[mask, sensor_cols] = regime_scalers[regime_id].transform(df.loc[mask, sensor_cols]).astype(np.float32)

        
        regime_dummies = pd.get_dummies(df['regime'], prefix='regime').astype(int)

            # ensure all regime columns exist
        for i in range(6):
            col = f'regime_{i}'
            if col not in regime_dummies.columns:
                regime_dummies[col] = 0

        # keep column order consistent
        regime_dummies = regime_dummies[
            sorted(regime_dummies.columns)
        ]

        df = pd.concat([df, regime_dummies], axis=1)

        df.drop('regime', axis=1, inplace=True)

        # select features
        X = df[feature_cols]



        # CREATE SEQUENCES


        X = X.values.astype(np.float32)
        # repeat to create sequence
        sequence = np.repeat(
            X[:, np.newaxis, :],
            sequence_length,
            axis=1
        )

        
        # PREDICT
        

        predictions = model.predict(sequence, verbose=0)

        # final prediction
        final_rul = float(predictions[0][0])
        return {"prediction": final_rul}