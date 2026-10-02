import streamlit as st
import pandas as pd
import joblib

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Simulasi Predictive Maintenance", page_icon="⚙️", layout="wide")

# 2. LOAD MODEL DAN SCALER
@st.cache_resource
def load_models():
    # Memuat model Random Forest dan Scaler dari folder lokal
    rf_model = joblib.load('models/milling_machine_rf_model.pkl')
    scaler = joblib.load('models/milling_scaler.pkl')
    return rf_model, scaler

model, scaler = load_models()

# 3. ANTARMUKA PENGGUNA (UI)
st.title("⚙️ Simulasi Predictive Maintenance: Mesin Frais")
st.markdown("""
Aplikasi ini merupakan prototipe **Machine Learning (Random Forest)** untuk mensimulasikan 
klasifikasi kondisi operasional (*Machine failure*) pada Mesin Frais (*Milling Machine*). 

*Proyek ini merupakan bentuk penerapan inovasi teknologi berbasis data pada sektor industri (**SDG 9 - Target 9.5**).*
""")

st.sidebar.header("Input Data Operasional")

# Form Input Sensor
tipe_mesin = st.sidebar.selectbox("Kualitas Produk (Type)", ["Low (L)", "Medium (M)", "High (H)"])
air_temp = st.sidebar.number_input("Air temperature [K]", min_value=290.0, max_value=310.0, value=298.1, step=0.1)
process_temp = st.sidebar.number_input("Process temperature [K]", min_value=300.0, max_value=320.0, value=308.6, step=0.1)
rpm = st.sidebar.number_input("Rotational speed [rpm]", min_value=1000, max_value=3000, value=1551, step=1)
torque = st.sidebar.number_input("Torque [Nm]", min_value=10.0, max_value=80.0, value=42.8, step=0.1)
tool_wear = st.sidebar.number_input("Tool wear [min]", min_value=0, max_value=300, value=10, step=1)

# 4. PEMROSESAN FITUR
# Transformasi kategori menjadi One-Hot Encoding sesuai dengan format saat training
type_l = 1 if tipe_mesin == "Low (L)" else 0
type_m = 1 if tipe_mesin == "Medium (M)" else 0

# Menyusun DataFrame agar formatnya sama persis dengan X_train
input_data = pd.DataFrame([[air_temp, process_temp, rpm, torque, tool_wear, type_l, type_m]], 
                          columns=['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type_L', 'Type_M'])

st.subheader("Data Parameter Input (Simulasi)")
st.dataframe(input_data, hide_index=True)

# 5. EKSEKUSI PREDIKSI
if st.button("🔍 Jalankan Simulasi Prediksi", type="primary", use_container_width=True):
    # Melakukan standardisasi (scaling) pada data input
    input_scaled = scaler.transform(input_data)
    
    # Melakukan prediksi kelas target
    prediksi = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    st.subheader("Hasil Klasifikasi Machine Learning")
    
    # Menampilkan hasil klasifikasi berdasarkan output model
    if prediksi == 1:
        st.error("⚠️ **HASIL KLASIFIKASI: MESIN DIPREDIKSI MENGALAMI FAILURE**", icon="🚨")
        st.write(f"Tingkat Keyakinan Model: **{probabilitas[1] * 100:.2f}%**")
        st.markdown("> *Terdapat indikasi anomali pada parameter fisika mesin berdasarkan data historis. Disarankan untuk melakukan pengecekan operasional lebih lanjut.*")
    else:
        st.success("✅ **HASIL KLASIFIKASI: MESIN DIPREDIKSI NORMAL**", icon="🛠️")
        st.write(f"Tingkat Keyakinan Model: **{probabilitas[0] * 100:.2f}%**")
        st.markdown("> *Kondisi operasional mesin frais diprediksi berada dalam batas wajar.*")