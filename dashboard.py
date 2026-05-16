import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ==========================================
# Konfigurasi Tampilan Halaman Streamlit
# ==========================================
st.set_page_config(page_title="E-Commerce Analytics Dashboard", page_icon="📦", layout="wide")
sns.set_theme(style="darkgrid")

# ==========================================
# Fungsi untuk Memuat dan Membersihkan Data
# Menggunakan @st.cache_data agar proses komputasi lebih cepat
# ==========================================
@st.cache_data
def load_and_clean_data():
    # Load dataset
    orders_df = pd.read_csv("orders_dataset.csv")
    reviews_df = pd.read_csv("order_reviews_dataset.csv")
    
    # Cleaning data orders
    datetime_cols = ['order_purchase_timestamp', 'order_approved_at', 
                     'order_delivered_carrier_date', 'order_delivered_customer_date', 
                     'order_estimated_delivery_date']
    
    for col in datetime_cols:
        orders_df[col] = pd.to_datetime(orders_df[col])
        
    orders_cleaned = orders_df[orders_df['order_status'] == 'delivered'].copy()
    orders_cleaned.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'], inplace=True)
    
    # Menggabungkan data orders dan reviews
    main_df = pd.merge(orders_cleaned, reviews_df, on="order_id", how="inner")
    
    # Feature Engineering (Menyiapkan variabel komputasi)
    # 1. Menghitung status delay
    main_df['is_delayed'] = main_df['order_delivered_customer_date'] > main_df['order_estimated_delivery_date']
    main_df['delivery_status'] = main_df['is_delayed'].map({False: 'Tepat Waktu/Lebih Cepat', True: 'Terlambat'})
    
    # 2. Menghitung selisih hari untuk clustering
    main_df['delivery_deviation_days'] = (main_df['order_delivered_customer_date'] - main_df['order_estimated_delivery_date']).dt.days
    
    def categorize_delivery(deviation):
        if deviation < 0:
            return "Lebih Cepat"
        elif deviation == 0:
            return "Tepat Waktu"
        elif deviation > 0 and deviation <= 3:
            return "Terlambat Ringan (1-3 Hari)"
        else:
            return "Terlambat Parah (> 3 Hari)"
            
    main_df['delivery_cluster'] = main_df['delivery_deviation_days'].apply(categorize_delivery)
    
    # 3. Ekstraksi waktu (Hari dan Jam)
    main_df['purchase_day'] = main_df['order_purchase_timestamp'].dt.day_name()
    main_df['purchase_hour'] = main_df['order_purchase_timestamp'].dt.hour
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    main_df['purchase_day'] = pd.Categorical(main_df['purchase_day'], categories=days_order, ordered=True)
    
    return main_df

# Memanggil fungsi data
all_df = load_and_clean_data()

# ==========================================
# Sidebar (Menu Navigasi dan Filter)
# ==========================================
with st.sidebar:
    st.title("📦 E-Commerce Analytics")
    st.markdown("Dashboard ini menganalisis performa logistik dan pola transaksi.")
    
    # Filter interaktif berdasarkan tahun
    min_year = all_df['order_purchase_timestamp'].dt.year.min()
    max_year = all_df['order_purchase_timestamp'].dt.year.max()
    
    selected_year = st.selectbox(
        label="Pilih Tahun Analisis:",
        options=["Semua Tahun"] + list(range(min_year, max_year + 1))
    )

# Menerapkan filter tahun pada data utama
if selected_year != "Semua Tahun":
    filtered_df = all_df[all_df['order_purchase_timestamp'].dt.year == selected_year]
else:
    filtered_df = all_df

# ==========================================
# Main Dashboard (Visualisasi)
# ==========================================
st.title("E-Commerce Public Performance Dashboard 📊")
st.markdown("---")

# Row 1: Menampilkan Metrik Utama (KPI)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Pesanan Valid", value=f"{filtered_df['order_id'].nunique():,}")
with col2:
    avg_score = filtered_df['review_score'].mean()
    st.metric("Rata-Rata Kepuasan", value=f"{avg_score:.2f} / 5.00")
with col3:
    delay_rate = (filtered_df['is_delayed'].sum() / len(filtered_df)) * 100
    st.metric("Tingkat Keterlambatan", value=f"{delay_rate:.2f}%")

st.markdown("---")

# Row 2: Pertanyaan 1 & Analisis Lanjutan (Korelasi Logistik)
st.subheader("Dampak Performa Logistik Terhadap Kepuasan Pelanggan")
col_fig1, col_fig2 = st.columns(2)

with col_fig1:
    # Grafik 1: Tepat Waktu vs Terlambat
    st.markdown("**Perbandingan Keterlambatan secara Umum**")
    score_by_delay = filtered_df.groupby('delivery_status')['review_score'].mean().reset_index()
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.barplot(x='delivery_status', y='review_score', data=score_by_delay, 
                palette=["#2b9348", "#d90429"], ax=ax1)
    ax1.set_xlabel(None)
    ax1.set_ylabel("Rata-Rata Review Score")
    ax1.set_ylim(0, 5.2)
    
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    
    st.pyplot(fig1)

with col_fig2:
    # Grafik 2: Clustering Keterlambatan
    st.markdown("**Clustering Toleransi Keterlambatan (Binning)**")
    cluster_order = ["Lebih Cepat", "Tepat Waktu", "Terlambat Ringan (1-3 Hari)", "Terlambat Parah (> 3 Hari)"]
    cluster_analysis = filtered_df.groupby('delivery_cluster')['review_score'].mean().reindex(cluster_order).reset_index()
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x='delivery_cluster', y='review_score', data=cluster_analysis, 
                palette=["#2ecc71", "#27ae60", "#f39c12", "#c0392b"], ax=ax2)
    ax2.set_xlabel(None)
    ax2.set_ylabel(None)
    ax2.set_ylim(0, 5.2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15)
    
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
        
    st.pyplot(fig2)

st.markdown("---")

# Row 3: Pertanyaan 2 (Heatmap Pola Transaksi)
st.subheader("Distribusi Spasial Waktu Transaksi (Heatmap)")
st.markdown("Visualisasi ini membantu menentukan jadwal optimal untuk peluncuran kampanye *marketing*.")

transaction_pattern = filtered_df.groupby(['purchase_day', 'purchase_hour']).size().unstack(fill_value=0)

fig3, ax3 = plt.subplots(figsize=(14, 5))
sns.heatmap(transaction_pattern, cmap='Blues', linewidths=.5, ax=ax3)
ax3.set_xlabel('Jam dalam Sehari (00:00 - 23:00)')
ax3.set_ylabel('Hari')
st.pyplot(fig3)

st.caption("Proyek Analisis Data Dicoding - Disusun dengan Streamlit")
# ==========================================
