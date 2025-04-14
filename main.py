from fastapi import FastAPI, UploadFile
import librosa
import numpy as np
import tensorflow as tf

app = FastAPI()
path_sample_h5 = "multiscale_residual_cnn.h5"
model = tf.keras.models.load_model(path_sample_h5)


@app.post("/predict")
async def predict(file: UploadFile):
    # wav 파일을 numpy array로
    contents = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(contents)
    y, sr = librosa.load("temp.wav", sr=16000)
    
    # 7초 맞추기 + 정규화
    y = y[:112000] if len(y) >= 112000 else np.pad(y, (0, 112000 - len(y)))
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    y = y.reshape((1, 112000, 1))

    # 예측
    pred = model.predict(y)[0][0]
    return {"normal": float(1 - pred), "stroke": float(pred)}
