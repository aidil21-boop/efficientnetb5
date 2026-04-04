import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import requests
import os

from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Deteksi LSD Sapi", page_icon="🐄")

st.title("🛡️ Deteksi Penyakit LSD Sapi")
st.caption("by Aidil Putra Samudra")
st.write("Aplikasi ini menggunakan arsitektur EfficientNet-B5 untuk mengidentifikasi penyakit Lumpy Skin Disease.")

# ===============================
# LINK MODEL (HUGGING FACE)
# ===============================
MODEL_URL = "https://huggingface.co/spaces/samudra19/efficientnetb5/resolve/main/model_lsd_sapi.keras"
MODEL_PATH = "model_lsd_sapi.keras"

# ===============================
# DOWNLOAD MODEL
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
# LOAD MODEL
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
# STATUS MODEL
# ===============================
if model is not None:
    st.success("Model berhasil dimuat dan siap digunakan")
else:
    st.error("Model gagal dimuat")

st.divider()

# ===============================
# LABEL
# ===============================
CLASS_NAMES = ['Sehat (Healthy)', 'Terinfeksi LSD (Lumpy Skin)']

# ===============================
# VALIDASI INPUT SEDERHANA
# ===============================
def is_valid_image(image):
    img = np.array(image)

    if np.std(img) < 25:
        return False

    mean_val = np.mean(img)
    if mean_val < 50 or mean_val > 200:
        return False

    return True

# ===============================
# PREDIKSI
# ===============================
def predict(image_data, model):
    image = image_data.convert("RGB")
    image = image.resize((456, 456))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    if prediction.shape[-1] == 1:
        prob = float(prediction[0][0])
        THRESHOLD = 0.5  

        if prob >= THRESHOLD:
            result = CLASS_NAMES[1]
            confidence = prob * 100
        else:
            result = CLASS_NAMES[0]
            confidence = (1 - prob) * 100
    else:
        result = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

    return result, confidence, image

# ===============================
# EXPANDER
# ===============================
with st.expander("📘 Penjelasan Model"):
    st.write("""
    Model ini menggunakan arsitektur EfficientNet-B5 yang merupakan bagian dari Convolutional Neural Network (CNN).
    Model dilatih untuk mengklasifikasikan citra kulit sapi menjadi dua kategori: sehat dan terinfeksi Lumpy Skin Disease (LSD).
    """)

with st.expander("📖 Panduan Penggunaan"):
    st.write("""
    - Gunakan gambar kulit sapi yang jelas
    - Fokus pada area yang menunjukkan gejala
    - Hindari gambar blur atau terlalu jauh
    """)

st.divider()

# ===============================
# UPLOAD GAMBAR
# ===============================
uploaded_file = st.file_uploader("Upload foto tekstur kulit sapi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # PREVIEW INPUT
    st.subheader("Preview Gambar")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if model is not None:
        if st.button("Analisis Gambar"):
            with st.spinner("Sedang mendiagnosis..."):

                # VALIDASI INPUT
                if not is_valid_image(image):
                    st.error("Gambar tidak sesuai (kemungkinan bukan kulit sapi).")
                    st.warning("Silakan upload gambar dengan tekstur kulit sapi yang jelas.")
                    st.stop()

                label, score, processed_image = predict(image, model)

                st.divider()

                # ===============================
                # HASIL ANALISIS
                # ===============================
                st.subheader("Hasil Analisis")

                st.image(processed_image, caption="Gambar yang dianalisis (456x456)", use_container_width=True)

                st.metric(label="Tingkat Keyakinan", value=f"{score:.2f}%")
                st.progress(int(score))

                if score < 60:
                    st.error("Gambar tidak valid atau model tidak yakin.")
                    st.warning("Silakan gunakan gambar yang lebih jelas.")
                else:
                    if "LSD" in label:
                        st.error(f"Hasil: {label}")
                        st.warning("Segera lakukan penanganan medis.")
                    else:
                        st.success(f"Hasil: {label}")

    else:
        st.warning("Model belum siap, silakan cek log error.")

st.divider()

# ===============================
# INFORMASI LSD
# ===============================
with st.expander("📚 Tentang Penyakit LSD"):
    st.write("""
    Lumpy Skin Disease (LSD) adalah penyakit virus pada sapi yang ditandai dengan munculnya benjolan pada kulit,
    demam, penurunan produksi susu, dan dapat menyebabkan kerugian ekonomi yang signifikan.
    """)

# ===============================
# DISCLAIMER
# ===============================
st.warning("Hasil prediksi ini hanya sebagai alat bantu dan tidak menggantikan diagnosis dokter hewan.")

st.divider()
st.caption("© 2026 - Sistem Deteksi LSD Sapi | Skripsi Teknik Elektro")
