import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Deteksi LSD Sapi", page_icon="🐄")

st.title("🛡️ Deteksi Penyakit LSD Sapi by TA aidil-samudra")
st.write("Aplikasi ini menggunakan arsitektur MobileNetV3 untuk mengidentifikasi penyakit Lumpy Skin Disease.")

@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model("model_lsd_sapi.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_my_model()

CLASS_NAMES = ['Sehat (Healthy)', 'Terinfeksi LSD (Lumpy Skin)']

def predict(image_data, model):
    image = image_data.convert("RGB")
    image = image.resize((224, 224))

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
