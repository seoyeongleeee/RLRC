import streamlit as st
import requests

st.title("🧠 뇌졸중 음성 분류 시스템")

uploaded_file = st.file_uploader("음성 파일(.wav)을 업로드하세요", type=["wav"])

if uploaded_file is not None:
    if st.button("Start Classification"):
        with st.spinner("모델이 예측 중입니다..."):
            # FastAPI 서버에 wav 파일 전송
            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": uploaded_file}
            )
            result = response.json()

        # 결과 출력
        st.success("✅ 예측 완료!")
        st.write(f"정상 확률: {result['normal'] * 100:.2f}%")
        st.write(f"뇌졸중 확률: {result['stroke'] * 100:.2f}%")

        final_result = "정상" if result['normal'] > result['stroke'] else "뇌졸중"
        st.markdown(f"### 🔍 최종 판별: **{final_result}** 입니다.")

        # test redeploy
