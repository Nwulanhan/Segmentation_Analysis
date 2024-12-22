import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score

# === Kelas Analisis Pelanggan ===
class AnalisisPelanggan:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.persiapan_data()

    def persiapan_data(self):
        """Pembersihan & Transformasi Data"""
        self.df.columns = self.df.columns.str.strip()
        self.df['Terakhir Aktif'] = pd.to_datetime(self.df['Terakhir Aktif'], errors='coerce')
        self.df['Mendaftar'] = pd.to_datetime(self.df['Mendaftar'], errors='coerce')
        kolom_numerik = ['Total Pengeluaran', 'Pesanan']
        for kolom in kolom_numerik:
            self.df[kolom] = pd.to_numeric(self.df[kolom], errors='coerce')
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(subset=kolom_numerik, inplace=True)

    def plot_top_kota_pelanggan(self):
        """Top 10 Kota Berdasarkan Jumlah Pelanggan"""
        top_cities_customers = self.df['Kota'].value_counts().head(10).reset_index()
        top_cities_customers.columns = ['Kota', 'Jumlah Pelanggan']
        fig = px.bar(top_cities_customers, x='Kota', y='Jumlah Pelanggan',
                     title="Top 10 Cities by Number of Customers",
                     labels={'Kota': 'City', 'Jumlah Pelanggan': 'Number of Customers'},
                     color='Jumlah Pelanggan', color_continuous_scale='Blues')
        st.plotly_chart(fig)

    def plot_top_kota_spending(self):
        """Top 10 Kota Berdasarkan Total Pengeluaran"""
        top_cities_spending = self.df.groupby('Kota')['Total Pengeluaran'].sum().nlargest(10).reset_index()
        fig = px.bar(top_cities_spending, x='Kota', y='Total Pengeluaran', title="Top 10 Cities by Total Spending",
                     labels={'Kota': 'City', 'Total Pengeluaran': 'Total Spending'},
                     color='Total Pengeluaran', color_continuous_scale='Blues')
        st.plotly_chart(fig)

    def plot_distribusi_pengeluaran(self):
        """Distribusi Total Pengeluaran"""
        fig = px.histogram(self.df, x='Total Pengeluaran', nbins=50, title="Distribusi Total Pengeluaran",
                           color_discrete_sequence=['green'], marginal="box")
        st.plotly_chart(fig)

    def plot_kategori_pelanggan(self):
        """Kategori Pelanggan"""
        kategori = pd.cut(self.df['Total Pengeluaran'],
                          bins=[0, 100000, 500000, np.inf],
                          labels=['Low Spender', 'Medium Spender', 'High Spender'])
        kategori_df = kategori.value_counts().reset_index()
        kategori_df.columns = ['Kategori', 'Jumlah']
        fig = px.pie(kategori_df, names='Kategori', values='Jumlah', title="Kategori Pelanggan",
                     color='Kategori', color_discrete_map={'Low Spender': 'orange', 
                                                           'Medium Spender': 'blue', 
                                                           'High Spender': 'green'})
        st.plotly_chart(fig)

    def plot_korelasi_pesanan_pengeluaran(self):
        """Korelasi Pesanan vs Total Pengeluaran"""
        fig = px.scatter(self.df, x='Pesanan', y='Total Pengeluaran', 
                         title="Korelasi Pesanan vs Pengeluaran", 
                         color='Total Pengeluaran', size='Total Pengeluaran', hover_data=['Kota'])
        st.plotly_chart(fig)

    def analisis_klaster_raw(self):
        """Klasterisasi Pelanggan (Raw Data)"""
        fitur = ['Total Pengeluaran', 'Pesanan']
        X = self.df[fitur]

        kmeans = KMeans(n_clusters=3, random_state=42)
        self.df['Klaster'] = kmeans.fit_predict(X)

        # Visualisasi Klaster
        fig = px.scatter(
            self.df, x='Total Pengeluaran', y='Pesanan', color='Klaster',
            title='Klasterisasi Pelanggan (Raw Data)',
            labels={'Total Pengeluaran': 'Total Pengeluaran', 'Pesanan': 'Jumlah Pesanan'},
            color_continuous_scale='Viridis',
            hover_data=['Kota']
        )
        st.plotly_chart(fig)

        # Rekomendasi Bisnis
        cluster_summary = self.df.groupby('Klaster')[['Total Pengeluaran', 'Pesanan']].mean()
        st.write("💡 **Rekomendasi Bisnis:**")
        for cluster, stats in cluster_summary.iterrows():
            st.write(f"Klaster {cluster}:")
            st.write(f"- Pengeluaran Rata-rata: Rp {stats['Total Pengeluaran']:.2f}")
            st.write(f"- Pesanan Rata-rata: {stats['Pesanan']:.2f}")
        st.write("1. Fokus pada Medium Spender untuk program loyalitas.")
        st.write("2. Berikan diskon dan reward khusus untuk High Spender.")
        st.write("3. Tingkatkan jumlah transaksi untuk Low Spender melalui promo menarik.")
        st.write("4. Optimalisasi strategi iklan berdasarkan klaster pelanggan.")

    def analisis_klaster_standardized(self):
        """Klasterisasi Pelanggan (Standardized Data)"""
        fitur = ['Total Pengeluaran', 'Pesanan']
        X = self.df[fitur]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        # Add cluster labels back to dataframe
        self.df['Klaster Standardized'] = labels

        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(X_scaled, labels)
        st.subheader("Silhouette Score")
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")

        # Visualisasi Klaster
        fig = px.scatter(
            x=X_scaled[:, 0], y=X_scaled[:, 1], color=labels.astype(str),
            title='Klasterisasi Pelanggan (Standardized Data)',
            labels={'x': 'Standardized Total Pengeluaran', 'y': 'Standardized Jumlah Pesanan'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig)

        # Rekomendasi Bisnis
        cluster_summary = self.df.groupby('Klaster Standardized')[['Total Pengeluaran', 'Pesanan']].mean()
        st.write("💡 **Rekomendasi Bisnis:**")
        for cluster, stats in cluster_summary.iterrows():
            st.write(f"Klaster {cluster}:")
            st.write(f"- Pengeluaran Rata-rata: Rp {stats['Total Pengeluaran']:.2f}")
            st.write(f"- Pesanan Rata-rata: {stats['Pesanan']:.2f}")
        st.write("1. Fokus pada Medium Spender untuk program loyalitas.")
        st.write("2. Berikan diskon dan reward khusus untuk High Spender.")
        st.write("3. Tingkatkan jumlah transaksi untuk Low Spender melalui promo menarik.")
        st.write("4. Optimalisasi strategi iklan berdasarkan klaster pelanggan.")

    def rfm_analysis(self):
        """RFM Analysis"""
        rfm = self.df.groupby('Nama pengguna').agg({
            'Terakhir Aktif': lambda x: (self.df['Terakhir Aktif'].max() - x.max()).days,
            'Pesanan': 'count',
            'Total Pengeluaran': 'sum'
        }).rename(columns={
            'Terakhir Aktif': 'Recency',
            'Pesanan': 'Frequency',
            'Total Pengeluaran': 'MonetaryValue'
        })
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=False, duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], 4, labels=False, duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['MonetaryValue'], 4, labels=False, duplicates='drop')
        rfm['Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        st.dataframe(rfm)
        return rfm

    def seasonal_promotions_analysis(self):
        """Seasonal Promotions Analysis"""
        self.df['Month'] = self.df['Mendaftar'].dt.month
        seasonal_trends = self.df.groupby('Month')['Total Pengeluaran'].sum().reset_index()
        fig = px.line(seasonal_trends, x='Month', y='Total Pengeluaran', title="Seasonal Spending Trends",
                      labels={'Month': 'Month', 'Total Pengeluaran': 'Total Spending'})
        st.plotly_chart(fig)

    def behavioral_trends_analysis(self):
        """Behavioral Trends Analysis"""
        fig = px.histogram(self.df, x='Pesanan', nbins=30, title="Order Frequency Distribution",
                           labels={'Pesanan': 'Number of Orders'}, marginal="box", color_discrete_sequence=['blue'])
        st.plotly_chart(fig)

    def churn_prediction(self):
        """Churn Prediction"""
        self.df['Terakhir Aktif'] = pd.to_datetime(self.df['Terakhir Aktif'], errors='coerce')
        self.df['Mendaftar'] = pd.to_datetime(self.df['Mendaftar'], errors='coerce')
        self.df = self.df.dropna(subset=['Terakhir Aktif', 'Mendaftar'])

        # Feature Engineering
        self.df['Recency'] = (self.df['Terakhir Aktif'].max() - self.df['Terakhir Aktif']).dt.days
        self.df['Lifetime'] = (self.df['Terakhir Aktif'] - self.df['Mendaftar']).dt.days
        self.df['AverageOrderValue'] = self.df['Total Pengeluaran'] / self.df['Pesanan']
        self.df['Churn'] = np.where(self.df['Recency'] > 90, 1, 0)

        # Drop missing or invalid values
        self.df = self.df.dropna()

        # Select relevant features
        features = ['Recency', 'Pesanan', 'Total Pengeluaran', 'Lifetime', 'AverageOrderValue']
        X = self.df[features]
        y = self.df['Churn']

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display results
        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Feature importance
        feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        fig = px.bar(feature_importances, x=feature_importances.values, y=feature_importances.index,
                     orientation='h', title="Feature Importance for Churn Prediction",
                     labels={'x': 'Importance', 'index': 'Feature'})
        st.plotly_chart(fig)

    def tampilkan_ringkasan(self):
        st.subheader("📊 Ringkasan Statistik Pelanggan")
        st.write(f"**Total Pelanggan:** {len(self.df)}")
        st.write(f"**Rata-rata Pengeluaran:** Rp {self.df['Total Pengeluaran'].mean():,.2f}")
        st.write(f"**Rata-rata Pesanan:** {self.df['Pesanan'].mean():.2f}")

# === Dashboard Streamlit ===
def main():
    st.set_page_config(page_title="Dashboard Analisis Pelanggan", layout="wide")
    st.title("📈 Dashboard Analisis Pelanggan Interaktif")
    
    # Sidebar untuk navigasi
    st.sidebar.title("Navigasi")
    pilihan = st.sidebar.radio("Pilih Analisis:", 
                               ["Ringkasan Statistik", 
                                "RFM Analysis",
                                "Top 10 Kota Berdasarkan Jumlah Pelanggan", 
                                "Top 10 Kota Berdasarkan Total Pengeluaran", 
                                "Distribusi Total Pengeluaran", 
                                "Kategori Pelanggan", 
                                "Korelasi Pesanan vs Pengeluaran", 
                                "Klasterisasi Pelanggan (Raw Data)",
                                "Klasterisasi Pelanggan (Standardized Data)",
                                "Seasonal Spending Trends",
                                "Behavioral Trends",
                                "Churn Prediction"])

    # Load data dan inisialisasi
    file_path = 'Updated_Customer_Report_With_Abbreviations.csv'
    analisis = AnalisisPelanggan(file_path)

    # Visualisasi interaktif
    if pilihan == "Ringkasan Statistik":
        analisis.tampilkan_ringkasan()
    elif pilihan == "RFM Analysis":
        st.subheader("📊 RFM Analysis")
        analisis.rfm_analysis()
    elif pilihan == "Top 10 Kota Berdasarkan Jumlah Pelanggan":
        st.subheader("🏙️ Top 10 Kota Berdasarkan Jumlah Pelanggan")
        analisis.plot_top_kota_pelanggan()
    elif pilihan == "Top 10 Kota Berdasarkan Total Pengeluaran":
        st.subheader("🏙️ Top 10 Kota Berdasarkan Total Pengeluaran")
        analisis.plot_top_kota_spending()
    elif pilihan == "Distribusi Total Pengeluaran":
        st.subheader("💰 Distribusi Total Pengeluaran")
        analisis.plot_distribusi_pengeluaran()
    elif pilihan == "Kategori Pelanggan":
        st.subheader("📊 Kategori Pelanggan")
        analisis.plot_kategori_pelanggan()
    elif pilihan == "Korelasi Pesanan vs Pengeluaran":
        st.subheader("🔗 Korelasi Pesanan vs Pengeluaran")
        analisis.plot_korelasi_pesanan_pengeluaran()
    elif pilihan == "Klasterisasi Pelanggan (Raw Data)":
        st.subheader("🔍 Klasterisasi Pelanggan (Raw Data)")
        analisis.analisis_klaster_raw()
    elif pilihan == "Klasterisasi Pelanggan (Standardized Data)":
        st.subheader("🔍 Klasterisasi Pelanggan (Standardized Data)")
        analisis.analisis_klaster_standardized()
    elif pilihan == "Seasonal Spending Trends":
        st.subheader("📅 Seasonal Spending Trends")
        analisis.seasonal_promotions_analysis()
    elif pilihan == "Behavioral Trends":
        st.subheader("📈 Behavioral Trends")
        analisis.behavioral_trends_analysis()
    elif pilihan == "Churn Prediction":
        st.subheader("🔄 Churn Prediction")
        analisis.churn_prediction()

if __name__ == "__main__":
    main()
