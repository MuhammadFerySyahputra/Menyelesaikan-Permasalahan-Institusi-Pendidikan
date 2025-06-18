import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Performa Mahasiswa", layout="wide")

# Fungsi load model pipeline
@st.cache_resource
def load_model(path='model/trained_pipeline.joblib'):
    return joblib.load(path)

pipeline = load_model()

# Ambil kolom dari dataset acuan
dataset_path = 'model/dataset_fitur_terbaik.csv'
df_example = pd.read_csv(dataset_path)
expected_columns = df_example.columns.tolist()

# Judul utama
st.markdown("""
    <h1 style='text-align: center; color: navy;'>🎓 Prediksi Performa Akademik Mahasiswa</h1>
    <p style='text-align: center; font-size: 18px;'>Upload file CSV dan prediksi apakah mahasiswa berada dalam kategori <b>'Dropout'</b> atau <b>'Graduate'</b>.</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Layout dua kolom
col1, col2 = st.columns([1.5, 2])

with col1:
    st.subheader("📂 1. Upload File CSV")
    uploaded_file = st.file_uploader("Unggah file dengan kolom yang sesuai:", type=["csv"])
    st.caption("Pastikan kolom sesuai dengan data training.")
    
with col2:
    st.subheader("🧾 Kolom yang Diharapkan")
    with st.expander("Klik untuk melihat daftar kolom yang dibutuhkan"):
        st.code(expected_columns)

# Jika ada file yang di-upload
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ File CSV tidak dapat dibaca. Error: {e}")
    else:
        if list(uploaded_df.columns) != expected_columns:
            st.error("⚠️ Kolom pada file CSV tidak sesuai dengan yang diharapkan oleh model.")
            st.text(f"✅ Diharapkan: {expected_columns}")
            st.text(f"📂 Diterima: {list(uploaded_df.columns)}")
        else:
            st.success("✅ Kolom sesuai. Data siap diproses!")
            st.subheader("👀 2. Preview Data")
            st.dataframe(uploaded_df.head(), use_container_width=True)

            if st.button("🚀 Jalankan Prediksi Sekarang"):
                try:
                    pred_proba = pipeline.predict_proba(uploaded_df)[:, 1]
                    pred_class = pipeline.predict(uploaded_df)

                    results_df = uploaded_df.copy()
                    results_df['Predicted_Probability'] = np.round(pred_proba, 4)
                    results_df['Predicted_Label'] = ['Dropout' if p == 1 else 'Graduate' for p in pred_class]

                    st.subheader("📊 3. Hasil Prediksi")
                    st.dataframe(results_df, use_container_width=True)

                    # Tombol download hasil
                    csv_result = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Hasil Prediksi",
                        data=csv_result,
                        file_name='prediksi_performa_mahasiswa.csv',
                        mime='text/csv'
                    )
                except Exception as e:
                    st.error(f"❗ Terjadi error saat melakukan prediksi: {e}")

else:
    st.info("📤 Silakan upload file CSV terlebih dahulu untuk memulai prediksi.")
