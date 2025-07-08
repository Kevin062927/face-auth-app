import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="얼굴 감지 앱", layout="centered")
st.title("📸 얼굴 감지 웹앱")

uploaded_file = st.file_uploader("얼굴이 나온 사진을 업로드하세요", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    # OpenCV는 BGR을 사용하므로 변환
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 얼굴 인식기 로드 (haarcascade 파일은 자동 다운로드됨)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 얼굴에 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="얼굴이 감지되었습니다", use_column_width=True)

    if len(faces) > 0:
        st.success(f"얼굴 {len(faces)}개 감지됨!")
    else:
        st.warning("얼굴이 감지되지 않았습니다.")
