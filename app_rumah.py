import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# =====================================================
# 1. PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Dashboard Harga Tanah",
    page_icon="🏠",
    layout="wide"
)

# =====================================================
# 2. CSS
# =====================================================
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    color: #0F172A;
    font-weight: 800;
}
.sub-header {
    font-size: 1.1rem;
    color: #64748B;
    margin-bottom: 25px;
}
.metric-card {
    background-color: #F8FAFC;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #3B82F6;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
.stButton>button {
    width: 100%;
    height: 3em;
    font-weight: bold;
    background-color: #3B82F6;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. DATA REFERENSI
# =====================================================
PROVINSI_LIST = [
    "Aceh","Sumatera Utara","Sumatera Barat","Riau","Kepulauan Riau",
    "Jambi","Sumatera Selatan","Bangka Belitung","Bengkulu","Lampung",
    "DKI Jakarta","Jawa Barat","Jawa Tengah","DI Yogyakarta","Jawa Timur",
    "Banten","Bali","Nusa Tenggara Barat","Nusa Tenggara Timur",
    "Kalimantan Barat","Kalimantan Tengah","Kalimantan Selatan",
    "Kalimantan Timur","Kalimantan Utara","Sulawesi Utara","Gorontalo",
    "Sulawesi Tengah","Sulawesi Barat","Sulawesi Selatan","Sulawesi Tenggara",
    "Maluku","Maluku Utara","Papua","Papua Barat","Papua Tengah",
    "Papua Pegunungan","Papua Selatan","Papua Barat Daya"
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

# =====================================================
# 4. LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("catboost_final.cbm")
    return model

model = load_model()

# =====================================================
# 5. PREPROCESSING (OPS I 2 – FINAL)
# =====================================================
def preprocess_input(df):
    df = df.copy()

    df['provinsi'] = df['provinsi'].str.lower().str.strip()

    for col, rank_list in ranking.items():
        mapping = {cat: len(rank_list)-i for i,cat in enumerate(rank_list)}
        df[col] = df[col].map(lambda x: mapping.get(x, np.median(list(mapping.values()))))

    df['log_luas_tanah'] = np.log1p(df['luas_tanah_m2'])

    df['aksesibilitas_index'] = (
        df['jarak_ke_pusat_kota']*0.5 +
        df['lebar_jalan_m']*0.3 +
        df['pencapaian_angkutan_umum']*0.2
    )

    df['lingkungan_index'] = (
        df['gengsi_lingkungan']*0.4 +
        df['fasilitas_sosial_umum']*0.3 +
        df['kondisi_jalan']*0.3
    )

    df['banjir_penalty'] = np.where(df['daerah_banjir']>=2, df['daerah_banjir']**2, 0)

    df['log_luas_x_jarak'] = np.log1p(df['luas_tanah_m2'] * df['jarak_ke_pusat_kota'])

    df['legalitas_score'] = df['legalitas_tanah']**1.2
    df['efektivitas_lahan'] = df['bentuk_tanah']*0.6 + df['frontage']*0.4

    df['log_harga_total'] = df['log_luas_tanah']

    df.drop(columns=['luas_tanah_m2'], inplace=True)
    return df

def align_features(df, model):
    X = pd.DataFrame(index=df.index)
    for col in model.feature_names_:
        X[col] = df[col] if col in df.columns else 0
    return X

# =====================================================
# 6. UI
# =====================================================
st.markdown('<div class="main-header">🏠 Estimasi Harga Tanah</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Prediksi Harga Pasar Wajar (CatBoost)</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    provinsi = st.selectbox("Provinsi", PROVINSI_LIST)
    luas = st.number_input("Luas Tanah (m²)", 10.0, 10000.0, 100.0)
    lebar = st.number_input("Lebar Jalan (m)", 0.0, 50.0, 6.0)
    jarak = st.selectbox("Jarak ke Pusat Kota", ranking["jarak_ke_pusat_kota"])
    jenis = st.selectbox("Jenis Informasi", ranking["jenis_informasi"])
    sumber = st.selectbox("Sumber Data", ranking["sumber_data"])

with col2:
    gengsi = st.selectbox("Gengsi Lingkungan", ranking["gengsi_lingkungan"])
    fasum = st.selectbox("Fasilitas Umum", ranking["fasilitas_sosial_umum"])
    angkot = st.selectbox("Angkutan Umum", ranking["pencapaian_angkutan_umum"])
    banjir = st.selectbox("Potensi Banjir", ranking["daerah_banjir"])
    kondisi = st.selectbox("Kondisi Jalan", ranking["kondisi_jalan"])
    perkerasan = st.selectbox("Perkerasan Jalan", ranking["perkerasan_jalan"])

if st.button("Hitung Estimasi Harga"):
    df_input = pd.DataFrame([{
        "provinsi": provinsi,
        "luas_tanah_m2": luas,
        "lebar_jalan_m": lebar,
        "jenis_informasi": jenis,
        "sumber_data": sumber,
        "perkerasan_jalan": perkerasan,
        "kondisi_jalan": kondisi,
        "jarak_ke_pusat_kota": jarak,
        "daerah_banjir": banjir,
        "fasilitas_sosial_umum": fasum,
        "pencapaian_angkutan_umum": angkot,
        "gengsi_lingkungan": gengsi,
        "legalitas_tanah": "SHM",
        "bentuk_tanah": "Empat Persegi Panjang",
        "frontage": "Normal",
        "topografi_tanah": "Datar",
        "kontur_elevasi_tanah_cm_diatas_jalan_dibawah_jalan": "Sama Jalan",
        "kondisi_tanah": "Matang"
    }])

    df_proc = preprocess_input(df_input)
    X = align_features(df_proc, model)

    y_log = model.predict(X)[0]
    harga_juta = np.expm1(y_log)
    harga_rp = harga_juta * 1_000_000

    st.markdown(f"""
    <div class="metric-card">
        <h3>Estimasi Harga:</h3>
        <h1>Rp {harga_rp:,.0f} / m²</h1>
        <p>Total Aset: Rp {harga_rp*luas:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)
