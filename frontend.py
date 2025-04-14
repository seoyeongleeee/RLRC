import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

st.title("🧠 뇌졸중 음성 분류 시스템")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("multiscale_residual_cnn.h5")

model = load_model()

uploaded_file = st.file_uploader("음성 파일(.wav)을 업로드하세요", type=["wav"])

if uploaded_file is not None:
    if st.button("Start Classification"):
        with st.spinner("모델이 예측 중입니다..."):
            # .wav 파일을 numpy array로 읽기
            y, sr = librosa.load(uploaded_file, sr=16000)
            y = y[:112000] if len(y) >= 112000 else np.pad(y, (0, 112000 - len(y)))
            y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
            y = y.reshape((1, 112000, 1))

            pred = model.predict(y)[0][0]
            result = {"normal": float(1 - pred), "stroke": float(pred)}

        # 결과 표시
        st.success("✅ 예측 완료!")
        st.write(f"정상 확률: {result['normal'] * 100:.2f}%")
        st.write(f"뇌졸중 확률: {result['stroke'] * 100:.2f}%")

        final_result = "정상" if result['normal'] > result['stroke'] else "뇌졸중"
        st.markdown(f"### 🔍 최종 판별: **{final_result}** 입니다.")
