import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import requests
import os
import pandas as pd

from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Klasifikasi LSD Sapi", page_icon="🐄")

st.title("🛡️ Klasifikasi Penyakit LSD Sapi")
st.caption("by Aidil Putra Samudra")
st.write("Aplikasi ini menggunakan arsitektur EfficientNet-B5 untuk mengidentifikasi penyakit Lumpy Skin Disease.")

# ===============================
# LINK MODEL
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
CLASS_NAMES = ['Sehat (Healthy)', 'Terindikasi LSD (Lumpy Skin)']

# ===============================
# VALIDASI INPUT
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
# LABEL CONFIDENCE
# ===============================
def confidence_label(score):
    if score >= 85:
        return "Sangat Tinggi"
    elif score >= 70:
        return "Tinggi"
    elif score >= 60:
        return "Cukup"
    else:
        return "Rendah"

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
        if prob >= 0.5:
            return CLASS_NAMES[1], prob * 100, image
        else:
            return CLASS_NAMES[0], (1 - prob) * 100, image
    else:
        return CLASS_NAMES[np.argmax(prediction)], float(np.max(prediction)) * 100, image

# ===============================
# PENJELASAN
# ===============================
with st.expander("📘 Penjelasan Model"):
    st.write("""
    Aplikasi ini menggunakan model Convolutional Neural Network (CNN) dengan arsitektur EfficientNet-B5 
    yang telah dilatih untuk mengklasifikasikan citra kulit sapi. Model bekerja dengan mengekstraksi fitur visual 
    seperti tekstur kulit, pola nodul, dan perubahan warna yang berkaitan dengan penyakit Lumpy Skin Disease (LSD). 

    EfficientNet-B5 dipilih karena memiliki kemampuan optimasi pada kedalaman, lebar, dan resolusi jaringan, 
    sehingga mampu menghasilkan akurasi yang lebih baik dibandingkan model CNN konvensional.
    """)

with st.expander("📖 Panduan Penggunaan"):
    st.write("""
   - Gunakan gambar kulit sapi yang jelas dan tidak blur
    - Pastikan pencahayaan cukup (tidak terlalu gelap atau terang)
    - Fokus pada area kulit yang menunjukkan gejala
    - Hindari penggunaan gambar dengan banyak objek atau latar belakang kompleks

    Batasan Sistem:
    - Sistem hanya dirancang untuk citra kulit sapi
    - Gambar selain sapi dapat menghasilkan prediksi yang tidak valid
    - Model hanya mengenali pola berdasarkan data latih, sehingga kondisi lapangan yang berbeda 
      dapat mempengaruhi hasil prediksi
    """)
with st.expander("⚙️ Cara Kerja Sistem"):
    st.write("""
    1. Gambar diunggah oleh pengguna
    2. Sistem melakukan preprocessing (resize dan normalisasi)
    3. Model EfficientNet-B5 mengekstraksi fitur dari citra
    4. Sistem melakukan klasifikasi (Sehat atau Terindikasi LSD)
    5. Hasil ditampilkan dalam bentuk label dan nilai confidence
    """)

st.caption("Catatan: Model hanya dilatih pada dataset tertentu, hasil dapat berbeda pada kondisi lapangan.")

st.divider()

# ===============================
# UPLOAD MULTIPLE
# ===============================
uploaded_files = st.file_uploader(
    "Upload maksimal 10 gambar kulit sapi...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.error("Maksimal 10 gambar!")
        st.stop()

    st.subheader("Preview Gambar")
    for i, file in enumerate(uploaded_files):
        st.image(Image.open(file), caption=f"Gambar {i+1}", use_container_width=True)

    if model is not None:
        if st.button("Analisis Semua Gambar"):

            results = []
            progress_bar = st.progress(0)

            for idx, file in enumerate(uploaded_files):
                image = Image.open(file)

                brightness = np.mean(np.array(image))
                texture = np.std(np.array(image))

                st.divider()
                st.markdown(f"## 🔍 Hasil Gambar {idx+1}")

                # ===============================
                # CEK VALIDASI
                # ===============================
                if not is_valid_image(image):

                    st.image(image, caption="Gambar Tidak Valid", use_container_width=True)

                    st.error("❌ Gambar tidak valid")
                    st.write(f"Brightness: {brightness:.2f}")
                    st.write(f"Texture: {texture:.2f}")

                    results.append(("Tidak Valid", 0))

                else:
                    label, score, processed_image = predict(image, model)

                    st.image(processed_image, caption="Gambar Dianalisis", use_container_width=True)

                    st.write(f"Brightness: {brightness:.2f}")
                    st.write(f"Texture: {texture:.2f}")

                    st.metric("Confidence", f"{score:.2f}%")
                    st.write(f"Level: {confidence_label(score)}")

                    st.progress(int(score))

                    if score < 60:
                        st.warning("Model tidak yakin")
                    else:
                        if "LSD" in label:
                            st.error(label)
                            st.warning("""
        ⚠️ Tindakan Disarankan:
        - Segera hubungi dokter hewan
        - Pisahkan sapi yang terindikasi dari kawanan untuk mencegah penyebaran
        - Lakukan observasi lanjutan terhadap gejala klinis
        """)
                        else:
                            st.success(label)

                    results.append((label, score))

                progress_bar.progress((idx + 1) / len(uploaded_files))

            # ===============================
            # RINGKASAN
            # ===============================
            sehat = sum(1 for r in results if "Sehat" in r[0])
            sakit = sum(1 for r in results if "LSD" in r[0])
            invalid = sum(1 for r in results if "Tidak Valid" in r[0])

            st.divider()
            st.markdown("## 📊 Ringkasan Keseluruhan")

            df = pd.DataFrame({
                "Kategori": ["Sehat", "LSD", "Tidak Valid"],
                "Jumlah": [sehat, sakit, invalid]
            })

            st.bar_chart(df.set_index("Kategori"))

            st.write(f"Sehat: {sehat}")
            st.write(f"LSD: {sakit}")
            st.write(f"Tidak Valid: {invalid}")

    else:
        st.warning("Model belum siap")

st.divider()

# ===============================
# INFO LSD
# ===============================
with st.expander("📚 Tentang Penyakit LSD"):
    st.write("Lumpy Skin Disease (LSD) adalah penyakit infeksius pada sapi yang disebabkan oleh virus dari genus Capripoxvirus, ditandai dengan munculnya benjolan (nodul) pada kulit, demam, penurunan nafsu makan, serta penurunan produksi susu dan kondisi tubuh. Penyakit ini menyebar terutama melalui gigitan serangga seperti lalat dan nyamuk, serta dapat menyebabkan kerugian ekonomi yang signifikan pada peternak. Meskipun jarang menyebabkan kematian, LSD berdampak serius pada kesehatan ternak dan produktivitas, sehingga memerlukan pengendalian melalui vaksinasi, pengendalian vektor, dan biosekuriti yang baik.")

st.warning("Hasil hanya sebagai alat bantu, bukan diagnosis medis.")

st.divider()
st.caption("© 2026 - Sistem Deteksi LSD Sapi | Skripsi Teknik Elektro")
