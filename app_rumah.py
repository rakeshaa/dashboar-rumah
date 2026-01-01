import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier

# =====================================================
# 1. PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Dashboard Harga Tanah",
    page_icon="🏠",
    layout="wide"
)

# =====================================================
# 2. CSS (Tampilan Premium - Teks Gelap)
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
.stButton>button:hover {
    background-color: #2563EB;
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
# 4. LOAD MODELS (CLF & REG)
# =====================================================
@st.cache_resource
def load_models():
    # Load Stage 1: Classifier
    clf = CatBoostClassifier()
    clf.load_model("stage1_classifier.cbm")
    
    # Load Stage 2: Regressor
    reg = CatBoostRegressor()
    reg.load_model("stage2_regressor.cbm")
    
    return clf, reg

try:
    clf, reg = load_models()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'stage1_classifier.cbm' dan 'stage2_regressor.cbm' ada di folder yang sama. Error: {e}")
    st.stop()

# =====================================================
# 5. PREPROCESSING & ALIGNMENT
# =====================================================
def preprocess_input(df):
    df = df.copy()

    # Pastikan format string lowercase
    if 'provinsi' in df.columns:
        df['provinsi'] = df['provinsi'].str.lower().str.strip()

    # Ordinal Encoding
    for col, rank_list in ranking.items():
        if col in df.columns:
            mapping = {cat: len(rank_list)-i for i,cat in enumerate(rank_list)}
            df[col] = df[col].map(lambda x: mapping.get(x, np.median(list(mapping.values()))))

    # Feature Engineering
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

    # Duplicate log luas sebagai log harga total (sesuai logika training sebelumnya)
    df['log_harga_total'] = df['log_luas_tanah']

    # Drop kolom raw yang sudah ditransformasi
    df.drop(columns=['luas_tanah_m2'], inplace=True)
    
    return df

def align_features(df, model):
    """Memastikan urutan kolom sama persis dengan yang diminta model"""
    X = pd.DataFrame(index=df.index)
    for col in model.feature_names_:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0 # Default value jika fitur hilang
    return X

# =====================================================
# 6. USER INTERFACE
# =====================================================
st.markdown('<div class="main-header">🏠 Estimasi Harga Tanah</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Prediksi Harga Pasar Wajar (Two-Stage Model)</div>', unsafe_allow_html=True)

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

# =====================================================
# 7. EKSEKUSI PREDIKSI
# =====================================================
if st.button("Hitung Estimasi Harga"):

    # 1. Tampung Input User
    df_input = pd.DataFrame([{
        "provinsi": provinsi.lower(), # Pastikan lowercase
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
        "legalitas_tanah": "SHM", # Default Value sesuai request
        "bentuk_tanah": "Empat Persegi Panjang",
        "frontage": "Normal",
        "topografi_tanah": "Datar",
        "kontur_elevasi_tanah_cm_diatas_jalan_dibawah_jalan": "Sama Jalan",
        "kondisi_tanah": "Matang"
    }])

    # 2. Preprocessing Data
    df_proc = preprocess_input(df_input)

    # 3. STAGE 1 – CLASSIFICATION (Menentukan Segmen)
    # Align features khusus untuk Classifier
    X_stage1 = align_features(df_proc, clf)
    
    # Predict Segment
    segment_pred = clf.predict(X_stage1)[0]
    
    # Masukkan hasil prediksi segmen ke dalam dataframe untuk Stage 2
    # Catatan: Kolom 'segment_pred' ini harus ada di Stage 2
    df_proc['segment_pred'] = segment_pred 
    # Jika outputnya array numpy (misal ['High']), ambil elemen stringnya
    if isinstance(segment_pred, (np.ndarray, list)):
        df_proc['segment_pred'] = segment_pred[0]


    # 4. STAGE 2 – REGRESSION (Menentukan Harga)
    # Align features khusus untuk Regressor (sekarang sudah ada kolom segment_pred)
    X_stage2 = align_features(df_proc, reg)
    
    # Predict Log Harga
    y_log = reg.predict(X_stage2)[0]

    # Transformasi Balik (Log -> Harga Asli)
    harga_juta = np.expm1(y_log)
    harga_rp = harga_juta * 1_000_000

    # 5. Output Tampilan
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color:#64748B; margin:0;">Estimasi Harga Tanah</h3>
        <h1 style="color:#0F172A; margin:0;">Rp {harga_rp:,.0f} / m²</h1>
        <hr style="margin: 10px 0; border-top: 1px solid #E2E8F0;">
        <p style="color:#334155; margin:0;">
            Segmen Klasifikasi: <b>{df_proc['segment_pred'].iloc[0]}</b><br>
            Total Aset: <b>Rp {harga_rp*luas:,.0f}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
