import gradio as gr
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib


# LOAD MODEL + OBJECTS

model = tf.keras.models.load_model("my_lstm_model_w128_b16.keras")

ops_scaler = joblib.load("ops_scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
sequence_length = joblib.load("sequence_length.pkl")
regime_scalers = joblib.load("regime_scalers.pkl")
sensor_cols = joblib.load("sensor_cols.pkl")

_kmeans_raw = joblib.load("kmeans.pkl")
kmeans_centers = _kmeans_raw.cluster_centers_.astype(np.float32)  # extract just the numpy array


# PREDICTION FUNCTION

def predict_rul(op1,op2, op3, s2, s3, s4, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s20, s21):
    
    # read data
    df = pd.DataFrame([{
        "op_setting_1": op1,
        "op_setting_2": op2,
        "op_setting_3": op3,
        "sensor_2": s2,
        "sensor_3": s3,
        "sensor_4": s4,
        "sensor_6": s6,
        "sensor_7": s7,
        "sensor_8": s8,
        "sensor_9": s9,
        "sensor_10": s10,
        "sensor_11": s11,
        "sensor_12": s12,
        "sensor_13": s13,
        "sensor_14": s14,
        "sensor_15": s15,
        "sensor_16": s16,
        "sensor_17": s17,
        "sensor_20": s20,
        "sensor_21": s21
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
    return final_rul

app = gr.Interface(
    fn=predict_rul,
    inputs=[
        gr.Number(label="Operation Setting 1"),
        gr.Number(label="Operation Setting 2"),
        gr.Number(label="Operation Setting 3"),
        gr.Number(label="Sensor 2"),
        gr.Number(label="Sensor 3"),
        gr.Number(label="Sensor 4"),
        gr.Number(label="Sensor 6"),
        gr.Number(label="Sensor 7"),
        gr.Number(label="Sensor 8"),
        gr.Number(label="Sensor 9"),
        gr.Number(label="Sensor 10"),
        gr.Number(label="Sensor 11"),
        gr.Number(label="Sensor 12"),
        gr.Number(label="Sensor 13"),
        gr.Number(label="Sensor 14"),
        gr.Number(label="Sensor 15"),
        gr.Number(label="Sensor 16"),
        gr.Number(label="Sensor 17"),
        gr.Number(label="Sensor 20"),
        gr.Number(label="Sensor 21")
    ],
    outputs=gr.Number(label="Predicted RUL"),
    title="Turbofan Engine RUL Prediction",
    description="Enter operational settings and sensor values to estimate Remaining Useful Life."
)

app.launch()
