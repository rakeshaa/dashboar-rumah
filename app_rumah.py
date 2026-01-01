import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool

# ============================================
# 1. KONFIGURASI HALAMAN & CSS
# ============================================
st.set_page_config(
    page_title="Dashboard Harga Tanah",
    page_icon="🏠",
    layout="wide"
)

# Styling CSS untuk Tampilan Premium
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #0F172A;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-top: -10px;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        background-color: #3B82F6;
        color: white;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 2. DATA REFERENSI (LOGIKA TETAP)
# ============================================
PROVINSI_LIST = [
    "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Kepulauan Riau",
    "Jambi", "Sumatera Selatan", "Bangka Belitung", "Bengkulu", "Lampung",
    "DKI Jakarta", "Jawa Barat", "Jawa Tengah", "DI Yogyakarta",
    "Jawa Timur", "Banten", "Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur",
    "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan",
    "Kalimantan Timur", "Kalimantan Utara", "Sulawesi Utara", "Gorontalo",
    "Sulawesi Tengah", "Sulawesi Barat", "Sulawesi Selatan", "Sulawesi Tenggara",
    "Maluku", "Maluku Utara", "Papua", "Papua Barat", "Papua Tengah",
    "Papua Pegunungan", "Papua Selatan", "Papua Barat Daya"
]

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
# 3. FUNGSI LOGIKA (TIDAK BERUBAH)
# ============================================

@st.cache_resource
def load_model():
    try:
        model = CatBoostRegressor()
        model.load_model("catboost_final.cbm")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def align_features(df_input, model):
    required_features = model.feature_names_
    df_aligned = pd.DataFrame()
    for feature in required_features:
        if feature in df_input.columns:
            df_aligned[feature] = df_input[feature]
        else:
            df_aligned[feature] = 0 
    return df_aligned

def preprocess_input(df_input):
    df = df_input.copy()
    if 'provinsi' in df.columns:
        df['provinsi'] = df['provinsi'].str.lower().str.strip()
    
    for col, rank_list in ranking.items():
        if col in df.columns:
            ordinal_map = {cat: len(rank_list) - i for i, cat in enumerate(rank_list)}
            df[col] = df[col].map(ordinal_map).fillna(len(rank_list) // 2)

    df['log_luas_tanah'] = np.log1p(df['luas_tanah_m2'])
    df['aksesibilitas_index'] = (df['jarak_ke_pusat_kota'] * 0.5 + df['lebar_jalan_m'] * 0.3 + df['pencapaian_angkutan_umum'] * 0.2)
    df['lingkungan_index'] = (df['gengsi_lingkungan'] * 0.4 + df['fasilitas_sosial_umum'] * 0.3 + df['kondisi_jalan'] * 0.3)
    df['banjir_penalty'] = df['daerah_banjir'].apply(lambda x: x**2 if x >= 2 else 0)
    df['luas_x_jarak'] = df['luas_tanah_m2'] * df['jarak_ke_pusat_kota']
    df['log_luas_x_jarak'] = np.log1p(df['luas_x_jarak'])
    df['legalitas_score'] = df['legalitas_tanah'] ** 1.2
    df['efektivitas_lahan'] = (df['bentuk_tanah'] * 0.6 + df['frontage'] * 0.4)
    
    return df

# ============================================
# 4. TAMPILAN UTAMA (DIPERBAIKI)
# ============================================
model = load_model()

# Header Halaman
st.markdown('<p class="main-header">🏠 Estimasi Harga Tanah untuk RUmah Tempat Tinggal </p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prediksi Nilai Pasar Wajar menggunakan CatBoost Regression</p>', unsafe_allow_html=True)

if model:
    with st.container():
        # Menggunakan Tabs agar input terorganisir rapi
        tab1, tab2, tab3 = st.tabs([
            "📍 **Lokasi & Dimensi**", 
            "🏘️ **Lingkungan & Akses**", 
            "📄 **Legalitas & Fisik**"
        ])

        # --- TAB 1: Lokasi Dasar ---
        with tab1:
            col_t1_1, col_t1_2 = st.columns(2)
            with col_t1_1:
                provinsi = st.selectbox("Pilih Provinsi", PROVINSI_LIST)
                luas_tanah = st.number_input("Luas Tanah (m²)", 10.0, 10000.0, 100.0)
                lebar_jalan = st.number_input("Lebar Jalan Depan (m)", 0.0, 50.0, 6.0)
            with col_t1_2:
                jarak = st.selectbox("Jarak ke Pusat Kota", ranking["jarak_ke_pusat_kota"])
                jenis_info = st.selectbox("Jenis Informasi", ranking["jenis_informasi"]) 
                sumber_dat = st.selectbox("Sumber Data", ranking["sumber_data"]) 

        # --- TAB 2: Lingkungan ---
        with tab2:
            col_t2_1, col_t2_2 = st.columns(2)
            with col_t2_1:
                gengsi = st.selectbox("Gengsi Lingkungan", ranking["gengsi_lingkungan"])
                fasum = st.selectbox("Fasilitas Sosial/Umum", ranking["fasilitas_sosial_umum"])
                angkot = st.selectbox("Akses Angkutan Umum", ranking["pencapaian_angkutan_umum"])
            with col_t2_2:
                banjir = st.selectbox("Potensi Banjir", ranking["daerah_banjir"])
                kondisi_jln = st.selectbox("Kondisi Jalan", ranking["kondisi_jalan"])
                perkerasan = st.selectbox("Perkerasan Jalan", ranking["perkerasan_jalan"])

        # --- TAB 3: Fisik Detail & Legal ---
        with tab3:
            col_t3_1, col_t3_2 = st.columns(2)
            with col_t3_1:
                legal = st.selectbox("Legalitas Sertifikat", ranking["legalitas_tanah"])
                bentuk = st.selectbox("Bentuk Tanah", ranking["bentuk_tanah"])
                front = st.selectbox("Posisi Muka (Frontage)", ranking["frontage"])
            with col_t3_2:
                topo = st.selectbox("Topografi Tanah", ranking["topografi_tanah"])
                kontur = st.selectbox("Kontur Elevasi", ranking["kontur_elevasi_tanah_cm_diatas_jalan_dibawah_jalan"])
                kondisi_tnh = st.selectbox("Kondisi Pematangan", ranking["kondisi_tanah"])

    st.markdown("---")

    # ============================================
    # 5. EKSEKUSI & HASIL
    # ============================================
    c_btn, c_res = st.columns([1, 2])

    with c_btn:
        st.markdown("### 🔮 Eksekusi")
        st.write("Klik tombol di bawah untuk memproses prediksi.")
        predict_btn = st.button("Hitung Estimasi Harga")

    with c_res:
        if predict_btn:
            # A. Proses Prediksi
            data = {
                "provinsi": [provinsi], "luas_tanah_m2": [luas_tanah], "lebar_jalan_m": [lebar_jalan],
                "jenis_informasi": [jenis_info], "sumber_data": [sumber_dat], "perkerasan_jalan": [perkerasan],
                "kondisi_jalan": [kondisi_jln], "jarak_ke_pusat_kota": [jarak], "daerah_banjir": [banjir],
                "fasilitas_sosial_umum": [fasum], "pencapaian_angkutan_umum": [angkot], "gengsi_lingkungan": [gengsi],
                "legalitas_tanah": [legal], "bentuk_tanah": [bentuk], "frontage": [front],
                "topografi_tanah": [topo], "kontur_elevasi_tanah_cm_diatas_jalan_dibawah_jalan": [kontur],
                "kondisi_tanah": [kondisi_tnh]
            }
            
            df_raw = pd.DataFrame(data)
            df_processed = preprocess_input(df_raw)
            
            try:
                # Align & Predict
                df_final = align_features(df_processed, model)
                pool_data = Pool(df_final, cat_features=['provinsi'])
                pred_log = model.predict(pool_data)
                pred_val = np.expm1(pred_log[0]) * 1_000_000 
                
                # Tampilkan Hasil di Card
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin:0; color:#64748B; font-size:1rem;">Estimasi Harga Pasar Wajar:</h3>
                    <h1 style="margin:0; color:#0F172A; font-size:2.5rem;">Rp {pred_val:,.0f} <span style="font-size:1rem; color:#64748B">/ m²</span></h1>
                    <hr style="margin: 10px 0; border-top: 1px solid #E2E8F0;">
                    <p style="margin:0; color:#334155;"><strong>Estimasi Total Aset:</strong> Rp {pred_val * luas_tanah:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error("Terjadi kesalahan saat memproses prediksi.")
                st.exception(e)
        else:
            st.info("👈 Silakan lengkapi data dan klik tombol Hitung.")

    # ============================================
    # 6. FEATURE IMPORTANCE VISUALIZATION
    # ============================================
    st.markdown("### 📊 Analisis Faktor Harga")
    
    with st.expander("Lihat Grafik Feature Importance", expanded=False):
        try:
            # Data Processing
            feature_importance = model.get_feature_importance()
            feature_names = model.feature_names_
            
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False).head(15)

            # Plotting Clean Style
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.Blues(np.linspace(0.4, 1, len(fi_top := fi_df)))
            
            ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors)
            ax.invert_yaxis()
            
            # Remove clutter
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#CBD5E1')
            ax.tick_params(axis='x', colors='#64748B')
            ax.tick_params(axis='y', colors='#334155')
            ax.set_xlabel('Tingkat Signifikansi', color='#64748B')
            ax.set_title('Top 15 Faktor Penentu Harga', color='#0F172A', fontweight='bold')
            
            st.pyplot(fig)
            
        except Exception as e:
            st.warning("Visualisasi Feature Importance belum tersedia untuk model ini.")

else:
    st.error("🚨 File model 'catboost_final.cbm' tidak ditemukan. Harap upload file model ke direktori aplikasi.")

