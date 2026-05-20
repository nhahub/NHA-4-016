from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
import numpy as np
import librosa
import tensorflow_hub as hub

#Load model

model = tf.keras.models.load_model("fan_id06_100E_32B_yamnet_mfcc.keras")

#Apply same preprocessing

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
threshold = 0.19613205492496488

app = FastAPI()

class PredictionInput(BaseModel):
    path: str


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
async def predict(input: PredictionInput):
        path = input.path

        def split_audio(data, sr, window_sec=1.0):
            window_size = int(sr * window_sec)
            segments = []

            for i in range(0, len(data) - window_size, window_size):
                segments.append(data[i:i + window_size])

            return segments
        
        def extract_mfcc_segment(segment, sr=16000, n_mfcc=40):
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
            return np.mean(mfcc, axis=1)  # shape: (40,)

        def extract_yamnet_segment(segment):
            segment = segment.astype(np.float32)
            scores, embeddings, spectrogram = yamnet_model(segment)

            return np.mean(embeddings.numpy(), axis=0)  # shape: (1024,)
        
        def extract_combined_features(data, sr=16000):
            segments = split_audio(data, sr)

            combined_features = []

            for seg in segments:
                mfcc_feat = extract_mfcc_segment(seg, sr)
                yamnet_feat = extract_yamnet_segment(seg)

                combined = np.concatenate([mfcc_feat, yamnet_feat])
                combined_features.append(combined)

            return np.array(combined_features)  # shape: (num_segments, 1064)
        
        def apply_context(features, context_window=5):
            context_data = []

            for i in range(len(features) - context_window + 1):
                window = features[i:i + context_window].flatten()
                context_data.append(window)

            return np.array(context_data)
        
        def get_features(audio_path, sr=16000):
            data, _ = librosa.load(audio_path, sr=sr)
            data = librosa.util.normalize(data)

            features = extract_combined_features(data, sr)  # (N, 1064)

            features = apply_context(features, context_window=5)

            return features
        
        def majority_vote(preds, group_size):
            preds = np.array(preds)

            n_groups = len(preds) // group_size
            file_preds = []

            for i in range(n_groups):
                group = preds[i*group_size:(i+1)*group_size]
                vote = 1 if np.sum(group) > (group_size / 2) else 0
                file_preds.append(vote)

            return np.array(file_preds)

        
        # PREDICT
        
        features = get_features(path)
        predictions = model.predict(features, verbose=0)

        errors = np.mean((predictions - features) ** 2, axis=1)

        preds = (errors > threshold).astype(int)  # 1 = anomaly, 0 = normal
        majority_vote_preds = majority_vote(preds, 5)
        if len(majority_vote_preds) > 0 and majority_vote_preds[0] == 1:
            return {"prediction": "Anomaly detected!"}
        else:
            return {"prediction": "Normal sound."}
