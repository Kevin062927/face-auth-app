import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

st.title("🔐 얼굴 인증 시스템")

uploaded_image = st.file_uploader("📷 얼굴 사진을 업로드하세요", type=["jpg", "png"])

if uploaded_image:
    # 등록된 얼굴 불러오기
    known_image = face_recognition.load_image_file("known_faces/user.jpg")
    known_encoding = face_recognition.face_encodings(known_image)[0]

    # 사용자가 올린 얼굴 처리
    img = Image.open(uploaded_image)
    img_np = np.array(img)

    faces = face_recognition.face_encodings(img_np)

    if faces:
        result = face_recognition.compare_faces([known_encoding], faces[0])
        if result[0]:
            st.success("✅ 얼굴이 같아요! 인증 성공!")
        else:
            st.error("❌ 다른 사람이에요! 인증 실패!")
    else:
        st.warning("😕 얼굴을 찾지 못했어요.")
