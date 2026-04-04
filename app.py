import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import requests
import os

st.set_page_config(page_title="Deteksi LSD Sapi", page_icon="🐄")

st.title("🛡️ Deteksi Penyakit LSD Sapi")
st.write("by Aidil Putra Samudra")
st.write("Aplikasi ini menggunakan arsitektur EfficientNet-B5 untuk mengidentifikasi penyakit Lumpy Skin Disease.")

# ===============================
# 🔗 LINK MODEL HUGGING FACE
# ===============================
MODEL_URL = "https://huggingface.co/spaces/samudra19/efficientnetb5/resolve/main/model_lsd_sapi.keras"
MODEL_PATH = "model_lsd_sapi.keras"

# ===============================
# 📥 DOWNLOAD MODEL (sekali saja)
# ===============================
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model dari Hugging Face..."):
            try:
                r = requests.get(MODEL_URL, timeout=120)
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                st.error(f"Gagal download model: {e}")
                return False
    return True

# ===============================
# 🧠 LOAD MODEL (pakai cache)
# ===============================
@st.cache_resource
def load_my_model():
    success = download_model()
    if not success:
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_my_model()

# ===============================
# 🏷️ LABEL
# ===============================
CLASS_NAMES = ['Sehat (Healthy)', 'Terinfeksi LSD (Lumpy Skin)']

# ===============================
# 🔍 PREDIKSI
# ===============================
def predict(image_data, model):
    image = image_data.convert("RGB")
    image_size = 456
    image = image.resize((image_size, image_size))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction.shape[-1] == 1:
        prob = float(prediction[0][0])
        THRESHOLD = 0.70 

        if prob >= THRESHOLD:
            result = CLASS_NAMES[1] 
            confidence = prob * 100
        else:
            result = CLASS_NAMES[0]
            confidence = (1 - prob) * 100
    else:
        result = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

    return result, confidence

# ===============================
# 📤 UPLOAD GAMBAR
# ===============================
uploaded_file = st.file_uploader("Upload foto tekstur kulit sapi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto yang dianalisis", use_container_width=True)

    if model is not None:
        if st.button("Analisis Gambar"):
            with st.spinner("Sedang mendiagnosis..."):
                label, score = predict(image, model)

                st.write("---")

                if "LSD" in label:
                    st.error(f"Hasil: {label}")
                    st.warning(f"Tingkat Keyakinan: {score:.2f}%")
                    st.write("Saran: Segera hubungi dokter hewan dan pisahkan sapi dari kelompoknya.")
                else:
                    st.success(f"Hasil: {label}")
                    st.info(f"Tingkat Keyakinan: {score:.2f}%")
    else:
        st.warning("Model belum siap, silakan cek log error.")
