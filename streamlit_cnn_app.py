# streamlit_cnn_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os

# ---------------------------
# 모델 로드
# ---------------------------
MODEL_DIR = "saved_models"

def get_latest_model():
    """가장 최신 모델 파일을 찾아서 반환합니다."""
    # .h5 또는 .keras 확장자를 가진 파일 필터링
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith((".h5", ".keras"))]
    if not models:
        return None
    # 최신 모델을 위해 정렬 (보통 이름에 타임스탬프가 있다면 그것을 기준으로)
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

latest_model_path = get_latest_model()
model = tf.keras.models.load_model(latest_model_path) if latest_model_path else None

# ---------------------------
# 앱 UI
# ---------------------------
st.title("CNN 숫자 예측기 (MNIST)")
st.markdown("그림판에 **0~9** 숫자를 그리세요. 💡 **팁**: 숫자를 크고 굵게 그리고, 캔버스 중앙에 위치시켜주세요!")

# ---------------------------
# 그리기 캔버스 (선 굵기 조정)
# ---------------------------
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,  # 숫자를 더 굵게 그릴 수 있도록 선 굵기 증가
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ---------------------------
# 예측 버튼
# ---------------------------
if st.button("예측 실행") and canvas_result.image_data is not None and model:
    # ---------------------------
    # 이미지 처리 (전처리 개선)
    # ---------------------------
    # Streamlit Canvas에서 이미지 데이터 가져오기 (알파 채널 제거)
    img = canvas_result.image_data[:, :, 0]

    # PIL Image로 변환 및 28x28로 리사이징 (LANCZOS 알고리즘 사용으로 품질 향상)
    img_pil = Image.fromarray(img.astype("uint8"))
    img_pil = img_pil.resize((28, 28), Image.LANCZOS)
    
    # 색상 반전 (흰 바탕에 검은 글씨 -> 검은 바탕에 흰 글씨로)
    img_pil = ImageOps.invert(img_pil)

    # numpy 배열로 변환 및 정규화
    img_np = np.array(img_pil).astype("float32") / 255.0
    
    # 임계값을 적용하여 불필요한 노이즈 제거 및 대비 향상 (선택 사항, 필요시 조절)
    # MNIST 데이터는 보통 흑백 픽셀 값이 0 또는 1이므로, 0.1보다 작은 값은 0으로, 큰 값은 1로 처리
    img_np = np.where(img_np > 0.1, 1.0, 0.0)

    # CNN 입력 형태에 맞게 차원 재구성 (배치 크기 1, 높이 28, 너비 28, 채널 1)
    img_final = img_np.reshape(1, 28, 28, 1)

    # ---------------------------
    # 예측
    # ---------------------------
    pred = model.predict(img_final)
    pred_class = np.argmax(pred)

    # ---------------------------
    # 출력
    # ---------------------------
    st.subheader(f"예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])

elif not model:
    st.warning("모델이 없습니다. 먼저 학습된 모델을 'saved_models' 폴더에 저장해 주세요.")
