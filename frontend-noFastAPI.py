import streamlit as st
import numpy as np
import tensorflow as tf
import librosa

st.set_page_config(page_title="🧠 뇌졸중 분류 시스템", layout="centered")
st.title("🧠 뇌졸중 음성 분류 시스템")

# ✅ 모델 로드 (최초 1회 캐시)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("multiscale_residual_cnn.h5")
    return model

model = load_model()

uploaded_file = st.file_uploader("음성 파일(.wav)을 업로드하세요", type=["wav"])

if uploaded_file is not None:
    if st.button("Start Classification"):
        with st.spinner("모델이 예측 중입니다..."):
            # 📥 wav 로드
            y, sr = librosa.load(uploaded_file, sr=16000)

            # ⏱ 7초 길이 맞추기
            y = y[:112000] if len(y) >= 112000 else np.pad(y, (0, 112000 - len(y)))

            # 📏 정규화
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            # ✅ 모델 입력 형태로 변환
            y = y.reshape((1, 112000, 1))

            # 🔮 예측 수행
            pred = model.predict(y)[0][0]
            result = {"normal": float(1 - pred), "stroke": float(pred)}

        # ✅ 결과 출력
        st.success("✅ 예측 완료!")
        st.write(f"정상 확률: {result['normal'] * 100:.2f}%")
        st.write(f"뇌졸중 확률: {result['stroke'] * 100:.2f}%")

        final_result = "정상" if result['normal'] > result['stroke'] else "뇌졸중"
        st.markdown(f"### 🔍 최종 판별: **{final_result}** 입니다.")
