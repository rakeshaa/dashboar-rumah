import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Diperlukan untuk grafik
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
    st.error(f"Gagal memuat model. Pastikan file .cbm ada di folder yang sama. Error: {e}")
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
        "provinsi": provinsi.lower(), 
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
    X_stage1 = align_features(df_proc, clf)
    segment_pred = clf.predict(X_stage1)[0]
    
    # Masukkan hasil prediksi segmen ke dataframe
    df_proc['segment_pred'] = segment_pred 
    if isinstance(segment_pred, (np.ndarray, list)):
        df_proc['segment_pred'] = segment_pred[0]

    # 4. STAGE 2 – REGRESSION (Menentukan Harga)
    X_stage2 = align_features(df_proc, reg)
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

# =====================================================
# 8. FEATURE IMPORTANCE (TAMBAHAN)
# =====================================================
st.markdown("---")
st.markdown('<div class="sub-header">📊 Analisis Faktor Pengaruh (Feature Importance)</div>', unsafe_allow_html=True)

with st.expander("Lihat Grafik Detail"):
    
    # Tab untuk memilih model mana yang ingin dilihat
    tab_fi1, tab_fi2 = st.tabs(["Faktor Penentu Harga (Regresi)", "Faktor Penentu Segmen (Klasifikasi)"])
    
    def plot_fi(model, color_hex):
        # Ambil Feature Importance dari model CatBoost
        fi_df = pd.DataFrame({
            "Feature": model.feature_names_,
            "Importance": model.get_feature_importance()
        }).sort_values(by="Importance", ascending=False).head(15) # Top 15 Features

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color=color_hex)
        ax.invert_yaxis() # Fitur terpenting di atas
        
        # Styling Chart agar bersih
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#CBD5E1')
        ax.tick_params(axis='x', colors='#64748B')
        ax.tick_params(axis='y', colors='#334155')
        ax.set_xlabel("Tingkat Signifikansi", color='#64748B')
        
        return fig, fi_df

    # --- TAB 1: REGRESSOR ---
    with tab_fi1:
        st.caption("Grafik ini menunjukkan variabel apa saja yang paling mempengaruhi **Nominal Harga** (Model Tahap 2).")
        fig1, df1 = plot_fi(reg, "#3B82F6") # Warna Biru
        st.pyplot(fig1)
        # Opsional: Tampilkan tabel data
        if st.checkbox("Tampilkan Data Tabel (Regresi)"):
            st.dataframe(df1, use_container_width=True)

    # --- TAB 2: CLASSIFIER ---
    with tab_fi2:
        st.caption("Grafik ini menunjukkan variabel apa saja yang paling mempengaruhi **Penentuan Kelas Segmen** (Model Tahap 1).")
        fig2, df2 = plot_fi(clf, "#64748B") # Warna Abu-abu
        st.pyplot(fig2)
        # Opsional: Tampilkan tabel data
        if st.checkbox("Tampilkan Data Tabel (Klasifikasi)"):
            st.dataframe(df2, use_container_width=True)
