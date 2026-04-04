import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import requests
import os

from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Deteksi LSD Sapi", page_icon="🐄")

st.title("🛡️ Deteksi Penyakit LSD Sapi")
st.write("by Aidil Putra Samudra")
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
# LABEL
# ===============================
CLASS_NAMES = ['Sehat (Healthy)', 'Terinfeksi LSD (Lumpy Skin)']

# ===============================
# VALIDASI INPUT SEDERHANA
# ===============================
def is_valid_image(image):
    img = np.array(image)

    # cek variasi warna (tekstur)
    if np.std(img) < 25:
        return False

    # cek brightness (hindari terlalu gelap/terang)
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
# PENJELASAN MODEL
# ===============================
st.subheader("Tentang Model")
st.write("""
Model ini menggunakan arsitektur EfficientNet-B5 yang merupakan bagian dari Convolutional Neural Network (CNN).
Model dilatih untuk mengklasifikasikan citra kulit sapi menjadi dua kategori: sehat dan terinfeksi Lumpy Skin Disease (LSD).
""")

# ===============================
# PANDUAN
# ===============================
st.subheader("Panduan Penggunaan")
st.write("""
- Gunakan gambar kulit sapi yang jelas
- Fokus pada area yang menunjukkan gejala
- Hindari gambar blur atau terlalu jauh
""")

# ===============================
# UPLOAD GAMBAR
# ===============================
uploaded_file = st.file_uploader("Upload foto tekstur kulit sapi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto yang dianalisis", use_container_width=True)

    if model is not None:
        if st.button("Analisis Gambar"):
            with st.spinner("Sedang mendiagnosis..."):

        # VALIDASI TAMBAHAN
        if not is_valid_image(image):
            st.error("Gambar tidak sesuai (kemungkinan bukan kulit sapi).")
            st.warning("Silakan upload gambar dengan tekstur kulit sapi yang jelas.")
            st.stop()

        label, score, processed_image = predict(image, model)

                st.write("---")

                # tampilkan preprocessing
                st.image(processed_image, caption="Gambar setelah preprocessing (456x456)")

                # confidence
                st.subheader("Hasil Prediksi")
                st.progress(int(score))

                # ===============================
                # VALIDASI INPUT (TAMBAHAN PENTING)
                # ===============================
                if score < 60:
                    st.error("Gambar tidak valid atau bukan kulit sapi.")
                    st.warning("Model tidak cukup yakin untuk memberikan prediksi yang akurat.")
                    st.info("Silakan upload gambar kulit sapi yang lebih jelas dan sesuai.")
                
                else:
                    if "LSD" in label:
                        st.error(f"Hasil: {label}")
                        st.warning(f"Tingkat Keyakinan: {score:.2f}%")
                        st.write("Saran: Segera hubungi dokter hewan dan pisahkan sapi dari kelompoknya.")
                    else:
                        st.success(f"Hasil: {label}")
                        st.info(f"Tingkat Keyakinan: {score:.2f}%")

    else:
        st.warning("Model belum siap, silakan cek log error.")

# ===============================
# INFORMASI LSD
# ===============================
st.subheader("Tentang Penyakit LSD")
st.write("""
Lumpy Skin Disease (LSD) adalah penyakit virus pada sapi yang ditandai dengan munculnya benjolan pada kulit,
demam, penurunan produksi susu, dan dapat menyebabkan kerugian ekonomi yang signifikan pada peternak.
""")

# ===============================
# DISCLAIMER
# ===============================
st.warning("""
Hasil prediksi ini hanya sebagai alat bantu dan tidak menggantikan diagnosis dokter hewan profesional.
""")
