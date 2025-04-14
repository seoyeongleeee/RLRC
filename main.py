from fastapi import FastAPI, UploadFile
import numpy as np
import tensorflow as tf
import soundfile as sf
import io

app = FastAPI()

# ✅ 1. 서버 시작 시 모델을 미리 한 번만 로딩
@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model("multiscale_residual_cnn.h5")
    print("✅ 모델 로드 완료")

# ✅ 2. 예측 API
@app.post("/predict")
async def predict(file: UploadFile):
    # ✅ 3. 업로드된 파일을 메모리에서 바로 읽기
    contents = await file.read()
    audio_bytes = io.BytesIO(contents)
    y, sr = sf.read(audio_bytes)

    # ✅ 4. 7초 길이 맞추기 (padding or truncation)
    y = y[:112000] if len(y) >= 112000 else np.pad(y, (0, 112000 - len(y)))
    
    # ✅ 5. 정규화 (음성 데이터는 max 기준 정규화)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    # ✅ 6. 모델 입력 형태로 reshape
    y = y.reshape((1, 112000, 1))

    # ✅ 7. 예측
    pred = model.predict(y)[0][0]

    # ✅ 8. 결과 반환
    result = {"normal": float(1 - pred), "stroke": float(pred)}
    print(f"📊 예측결과: normal={result['normal']:.4f}, stroke={result['stroke']:.4f}")
    return result
