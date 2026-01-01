import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool

# ======================================================
# 1. PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Dashboard Harga Tanah",
    page_icon="🏠",
    layout="wide"
)

# ======================================================
# 2. REFERENSI (IDENTIK TRAINING)
# ======================================================

PROVINSI_LIST = [
    "Aceh","Sumatera Utara","Sumatera Barat","Riau","Kepulauan Riau","Jambi",
    "Sumatera Selatan","Bangka Belitung","Bengkulu","Lampung","DKI Jakarta",
    "Jawa Barat","Jawa Tengah","DI Yogyakarta","Jawa Timur","Banten","Bali",
    "Nusa Tenggara Barat","Nusa Tenggara Timur","Kalimantan Barat",
    "Kalimantan Tengah","Kalimantan Selatan","Kalimantan Timur",
    "Kalimantan Utara","Sulawesi Utara","Gorontalo","Sulawesi Tengah",
    "Sulawesi Barat","Sulawesi Selatan","Sulawesi Tenggara","Maluku",
    "Maluku Utara","Papua","Papua Barat","Papua Tengah","Papua Pegunungan",
    "Papua Selatan","Papua Barat Daya"
]

ranking = {
    "jenis_informasi": ["Pemilik","Transaksi","Penawaran","Agen","Lainnya","Tidak Diketahui"],
    "sumber_data": ["Pemilik","Agen Properti","Calo"],
    "perkerasan_jalan": ["Beton Hotmix","Aspal Penetrasi","Paving","Perkerasan Sirtu","Perkerasan Tanah"],
    "kondisi_jalan": ["Baik","Cukup Baik","Rusak","Rusak Berat"],
    "jarak_ke_pusat_kota": ["Kurang dari 500 m","500 - 1000 m","1000 - 2000 m","Lebih dari 2000 m"],
    "daerah_banjir": ["Tidak Banjir","Banjir Musiman","Selalu Banjir"],
    "fasilitas_sosial_umum": ["Lengkap","Standar","Tidak Ada"],
    "pencapaian_angkutan_umum": ["Baik","Cukup","Kurang","Tidak Ada"],
    "gengsi_lingkungan": ["Bergengsi","Standar","Kurang Bergengsi","Daerah Rawan"],
    "legalitas_tanah": ["SHM","SHGB","AJB","Girik","Hak Pakai","Lainnya"],
    "bentuk_tanah": ["Empat Persegi Panjang","Trapesium","Segitiga","Tidak Beraturan"],
    "frontage": ["Normal","Lebar Depan","Sempit"],
    "topografi_tanah": ["Datar","Terasering","Bergelombang","Miring"],
    "kontur_elevasi_tanah_cm_diatas_jalan_dibawah_jalan": ["Sama Jalan","Diatas Jalan","Dibawah Jalan"],
    "kondisi_tanah": ["Matang","Mentah","Tanah Sawah","Tanah Ladang","Tanah Rawa"]
}

# ======================================================
# 3. LOAD MODEL (NO TRAINING)
# ======================================================

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_final.cbm")
    return model

model = load_model()

# ======================================================
# 4. PREPROCESSING (STRICT IDENTIK TRAINING)
# ======================================================

def ordinal_encode(df):
    df = df.copy()
    for col, rank_list in ranking.items():
        mapping = {cat: len(rank_list) - i for i, cat in enumerate(rank_list)}
        df[col] = df[col].map(mapping)
    return df

def feature_engineering(df):
    df = df.copy()

    # provinsi HARUS tetap ada (categorical)
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

    # DROP RAW FEATURE (SESUAI TRAINING)
    df = df.drop(columns=["luas_tanah_m2"], errors="ignore")

    return df

# ======================================================
# 5. UI INPUT
# ======================================================

st.title("🏠 Estimasi Harga Tanah Ruko (FINAL – STRICT)")

with st.form("input_form"):
    provinsi = st.selectbox("Provinsi", PROVINSI_LIST)
    luas_tanah = st.number_input("Luas Tanah (m²)", 10.0, 10000.0, 100.0)
    lebar_jalan = st.number_input("Lebar Jalan (m)", 1.0, 50.0, 6.0)

    jarak = st.selectbox("Jarak ke Pusat Kota", ranking["jarak_ke_pusat_kota"])
    jenis_info = st.selectbox("Jenis Informasi", ranking["jenis_informasi"])
    sumber_data = st.selectbox("Sumber Data", ranking["sumber_data"])
    gengsi = st.selectbox("Gengsi Lingkungan", ranking["gengsi_lingkungan"])
    fasum = st.selectbox("Fasilitas Sosial/Umum", ranking["fasilitas_sosial_umum"])
    angkot = st.selectbox("Akses Angkutan Umum", ranking["pencapaian_angkutan_umum"])
    banjir = st.selectbox("Potensi Banjir", ranking["daerah_banjir"])
    kondisi_jalan = st.selectbox("Kondisi Jalan", ranking["kondisi_jalan"])
    perkerasan = st.selectbox("Perkerasan Jalan", ranking["perkerasan_jalan"])
    legalitas = st.selectbox("Legalitas", ranking["legalitas_tanah"])
    bentuk = st.selectbox("Bentuk Tanah", ranking["bentuk_tanah"])
    frontage = st.selectbox("Frontage", ranking["frontage"])
    topo = st.selectbox("Topografi", ranking["topografi_tanah"])
    kontur = st.selectbox(
        "Kontur Elevasi",
        ranking["kontur_elevasi_tanah_cm_diatas_jalan_dibawah_jalan"]
    )
    kondisi_tanah = st.selectbox("Kondisi Tanah", ranking["kondisi_tanah"])

    submit = st.form_submit_button("Hitung Estimasi Harga")

# ======================================================
# 6. PREDIKSI (FINAL & AMAN)
# ======================================================

if submit:
    df_raw = pd.DataFrame([{
        "provinsi": provinsi.lower().strip(),
        "luas_tanah_m2": luas_tanah,
        "lebar_jalan_m": lebar_jalan,
        "jenis_informasi": jenis_info,
        "sumber_data": sumber_data,
        "perkerasan_jalan": perkerasan,
        "kondisi_jalan": kondisi_jalan,
        "jarak_ke_pusat_kota": jarak,
        "daerah_banjir": banjir,
        "fasilitas_sosial_umum": fasum,
        "pencapaian_angkutan_umum": angkot,
        "gengsi_lingkungan": gengsi,
        "legalitas_tanah": legalitas,
        "bentuk_tanah": bentuk,
        "frontage": frontage,
        "topografi_tanah": topo,
        "kontur_elevasi_tanah_cm_diatas_jalan_dibawah_jalan": kontur,
        "kondisi_tanah": kondisi_tanah
    }])

    df_encoded = ordinal_encode(df_raw)
    df_fe = feature_engineering(df_encoded)

    # STRICT FEATURE MATCH (TIDAK BISA ERROR LAGI)
    X = df_fe[model.feature_names_]

    pool = Pool(X, cat_features=["provinsi"])
    y_pred_log = model.predict(pool)

    # BALIK KE RUPIAH
    harga_juta = np.expm1(y_pred_log[0])
    harga_rp = harga_juta * 1_000_000

    st.success("Prediksi berhasil")
    st.metric("Harga Tanah (Rp / m²)", f"Rp {harga_rp:,.0f}")
    st.metric("Estimasi Total Aset", f"Rp {harga_rp * luas_tanah:,.0f}")

# ======================================================
# 7. FEATURE IMPORTANCE
# ======================================================

with st.expander("Lihat Feature Importance"):
    fi = pd.DataFrame({
        "Feature": model.feature_names_,
        "Importance": model.get_feature_importance()
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(fi["Feature"], fi["Importance"])
    ax.invert_yaxis()
    st.pyplot(fig)
