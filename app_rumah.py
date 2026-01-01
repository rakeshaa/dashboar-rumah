# ============================================
# DASHBOARD RUMAH TINGGAL – CATBOOST (FINAL)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

st.set_page_config(
    page_title="Dashboard Prediksi Harga Rumah",
    layout="wide"
)

# ============================================
# RANKING KATEGORIK (SAMA DENGAN TRAINING)
# ============================================

ranking = {
    "jenis_informasi": ["Pemilik", "Transaksi", "Penawaran", "Agen", "Lainnya", "Tidak Diketahui"],
    "sumber_data": ["Pemilik", "Agen Properti", "Calo"],
    "perkerasan_jalan": ["Beton Hotmix", "Aspal Penetrasi", "Paving", "Perkerasan Sirtu", "Perkerasan Tanah"],
    "kondisi_jalan": ["Baik", "Cukup Baik", "Rusak", "Rusak Berat"],
    "jarak_ke_pusat_kota": ["Kurang dari 500 m", "500 - 1000 m", "1000 - 2000 m", "Lebih dari 2000 m"],
    "daerah_banjir": ["Tidak Banjir", "Banjir Musiman", "Selalu Banjir"],
    "fasilitas_sosial_umum": ["Lengkap", "Standar", "Tidak Ada"],
    "pencapaian_angkutan_umum": ["Baik", "Cukup", "Kurang", "Tidak Ada"],
    "gengsi_lingkungan": ["Bergengsi", "Standar", "Kurang Bergengsi", "Daerah Rawan"],
    "legalitas_tanah": ["SHM", "SHGB", "AJB", "Girik", "Hak Pakai", "Lainnya"],
    "bentuk_tanah": ["Empat Persegi Panjang", "Trapesium", "Segitiga", "Tidak Beraturan"],
    "frontage": ["Normal", "Lebar Depan", "Sempit"],
    "topografi_tanah": ["Datar", "Terasering", "Bergelombang", "Miring"],
    "kontur_elevasi_tanah_cm_diatas_jalan_dibawah_jalan": ["Sama Jalan", "Diatas Jalan", "Dibawah Jalan"],
    "kondisi_tanah": ["Matang", "Mentah", "Tanah Sawah", "Tanah Ladang", "Tanah Rawa"]
}

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_final.cbm")
    return model

model = load_model()

# ============================================
# FEATURE ENGINEERING (SAMA PERSIS)
# ============================================

def feature_engineering(df):
    df = df.copy()

    df["log_luas_tanah"] = np.log1p(df["luas_tanah_m2"])

    df["aksesibilitas_index"] = (
        df["jarak_ke_pusat_kota"] * 0.5 +
        df["lebar_jalan_m"] * 0.3 +
        df["pencapaian_angkutan_umum"] * 0.2
    )

    df["lingkungan_index"] = (
        df["gengsi_lingkungan"] * 0.4 +
        df["fasilitas_sosial_umum"] * 0.3 +
        df["kondisi_jalan"] * 0.3
    )

    df["banjir_penalty"] = np.where(
        df["daerah_banjir"] >= 2,
        df["daerah_banjir"] ** 2,
        0
    )

    df["log_luas_x_jarak"] = np.log1p(
        df["luas_tanah_m2"] * df["jarak_ke_pusat_kota"]
    )

    df["legalitas_score"] = df["legalitas_tanah"] ** 1.2

    df["efektivitas_lahan"] = (
        df["bentuk_tanah"] * 0.6 +
        df["frontage"] * 0.4
    )

    df.drop(columns=["luas_tanah_m2"], inplace=True)

    return df

# ============================================
# UI
# ============================================

st.title("🏠 Dashboard Prediksi Harga Rumah Tinggal")
st.subheader("CatBoost Regressor")
st.markdown("---")

user_input = {}

# ============================================
# NUMERIK
# ============================================

st.subheader("Variabel Numerik")

col1, col2 = st.columns(2)

with col1:
    user_input["luas_tanah_m2"] = st.number_input("Luas Tanah (m²)", min_value=1.0)

with col2:
    user_input["lebar_jalan_m"] = st.number_input("Lebar Jalan (m)", min_value=1.0)

# ============================================
# KATEGORIK (DROPDOWN)
# ============================================

st.subheader("Variabel Kategorik")

cols = st.columns(3)
i = 0

for col, options in ranking.items():
    with cols[i % 3]:
        pilihan = st.selectbox(col.replace("_", " ").title(), options)
        user_input[col] = len(options) - options.index(pilihan)
    i += 1

# ============================================
# PREDIKSI
# ============================================

if st.button("🔮 Prediksi Harga"):
    df_user = pd.DataFrame([user_input])
    df_fe = feature_engineering(df_user)

    y_log = model.predict(df_fe)[0]
    y = np.expm1(y_log)

    st.success("Prediksi berhasil!")
    st.metric("Harga Rumah (Rp)", f"{y * 1_000_000:,.0f}")

# ============================================
# FEATURE IMPORTANCE
# ============================================

st.markdown("---")
st.subheader("Feature Importance")

fi = pd.DataFrame({
    "Variabel": model.feature_names_,
    "Importance": model.get_feature_importance()
}).sort_values("Importance", ascending=False)

st.dataframe(fi, use_container_width=True)

fig, ax = plt.subplots(figsize=(6,4))
ax.barh(fi["Variabel"], fi["Importance"])
ax.invert_yaxis()
ax.set_title("Feature Importance – Rumah Tinggal")
st.pyplot(fig)
