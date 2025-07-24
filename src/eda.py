import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    # Load the data
    try:
        df = pd.read_csv('Hotel_Reservations.csv')
    except FileNotFoundError:
        st.error("Error: Hotel_Reservations.csv not found. Make sure the file is in the same directory or provide the correct path.")
        st.stop()

    st.title('Exploratory Data Analysis of Hotel Reservations')

    # Display the dataframe
    st.header('Dataframe')
    st.dataframe(df.head())

    # Display the shape of the dataframe
    st.header('Shape of Dataframe')
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    # Display descriptive statistics
    st.header('Descriptive Statistics')
    st.dataframe(df.describe().T)

    # Exploratory Data Analysis Visualizations

    # 1. Distribusi Status Pemesanan
    st.header('1. Distribusi Status Pemesanan')
    st.write("Visualisasi ini menunjukkan distribusi antara reservasi yang dibatalkan (Canceled) dan tidak dibatalkan (Not_anceled).")
    fig_booking_status, ax_booking_status = plt.subplots(figsize=(8, 6))
    sns.countplot(x='booking_status', data=df, ax=ax_booking_status)
    ax_booking_status.set_title('Distribusi Status Pemesanan')
    ax_booking_status.set_xlabel('Status Pemesanan')
    ax_booking_status.set_ylabel('Jumlah')
    st.pyplot(fig_booking_status)
    st.write("Proporsi Status Pemesanan:")
    st.write(df['booking_status'].value_counts(normalize=True))
    jumlah_cancelled = df[df['booking_status'] == 'Canceled'].shape[0]
    jumlah_not_cancelled = df[df['booking_status'] == 'Not_Canceled'].shape[0]
    st.write(f"Jumlah baris dengan status 'Canceled': {jumlah_cancelled}")
    st.write(f"Jumlah baris dengan status 'Not_Canceled': {jumlah_not_cancelled}")
    st.write("Tingkat pembatalan yang tinggi (33%) menunjukkan masalah signifikan bagi manajemen hotel, seperti potensi kehilangan pendapatan dan tantangan dalam pengelolaan kapasitas kamar. Ini menegaskan pentingnya model prediksi pembatalan untuk mengidentifikasi reservasi berisiko tinggi.")

    # 2. Rata-rata lead time untuk pemesanan kamar hotel
    st.header('2. Rata-rata lead time untuk pemesanan kamar hotel')
    st.write("Histogram ini menunjukkan distribusi lead_time (waktu antara pemesanan dan tanggal kedatangan, dalam hari).")
    st.write(f"Average lead time: {df['lead_time'].mean()}")
    fig_lead_time, ax_lead_time = plt.subplots()
    df['lead_time'].hist(bins=30, ax=ax_lead_time)
    st.pyplot(fig_lead_time)
    st.write("Reservasi dengan lead time panjang mungkin lebih berisiko dibatalkan karena pelanggan memiliki lebih banyak waktu untuk mengubah rencana. Hotel dapat menawarkan insentif untuk mengurangi pembatalan pada reservasi jangka panjang.")

    # 3. Pemesanan Per Bulan
    st.header('3. Pemesanan Per Bulan')
    st.write("Bar plot ini menunjukkan jumlah reservasi per bulan.")
    fig_arrival_month, ax_arrival_month = plt.subplots()
    df['arrival_month'].value_counts().sort_index().plot(kind='bar', title='Bookings per Month', ax=ax_arrival_month)
    st.pyplot(fig_arrival_month)
    st.write("Puncak reservasi menunjukkan periode hari besar dimana pembatalan dapat berdampak besar pada pendapatan di bulan-bulan tersebut.")

    # 4. Kaitan Lead Time dan Cancellation
    st.header('4. Kaitan Lead Time dan Cancellation')
    st.write("Boxplot ini menunjukkan bahwa reservasi yang dibatalkan memiliki lead time yang lebih panjang dibandingkan yang tidak dibatalkan.")
    fig_lead_time_cancellation, ax_lead_time_cancellation = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='booking_status', y='lead_time', data=df, ax=ax_lead_time_cancellation)
    ax_lead_time_cancellation.set_title('Distribusi Lead Time berdasarkan Status Pemesanan')
    st.pyplot(fig_lead_time_cancellation)
    st.write("Reservasi dengan lead time panjang adalah indikator risiko pembatalan. Hotel dapat menerapkan kebijakan seperti deposit lebih tinggi atau komunikasi proaktif untuk reservasi jangka panjang guna mengurangi pembatalan.")

    # 5. Bagaimana pengaruh jumlah permintaan khusus terhadap status pemesanan?
    st.header('5. Bagaimana pengaruh jumlah permintaan khusus terhadap status pemesanan?')
    st.write("Visualisasi ini menunjukkan bagaimana jumlah permintaan khusus (misalnya, kamar dengan pemandangan, tempat tidur tambahan) berhubungan dengan status pemesanan.")
    fig_special_requests, ax_special_requests = plt.subplots(figsize=(10, 6))
    sns.countplot(x='no_of_special_requests', hue='booking_status', data=df, ax=ax_special_requests)
    ax_special_requests.set_title('Pengaruh Jumlah Permintaan Khusus terhadap Status Pemesanan')
    ax_special_requests.set_xlabel('Jumlah Permintaan Khusus')
    ax_special_requests.set_ylabel('Jumlah Pemesanan')
    ax_special_requests.legend(title='Status Pemesanan')
    st.pyplot(fig_special_requests)
    st.write("Ini mengindikasikan bahwa tamu yang membuat lebih sedikit permintaan khusus cenderung lebih berkomitmen pada pemesanan mereka. Hotel dapat mempertimbangkan untuk memberikan insentif atau layanan tambahan kepada tamu yang membuat permintaan khusus untuk meningkatkan loyalitas dan mengurangi pembatalan.")

    # 6. Bagaimana pengaruh jenis kamar yang dipesan terhadap status pemesanan?
    st.header('6. Bagaimana pengaruh jenis kamar yang dipesan terhadap status pemesanan?')
    st.write("Visualisasi ini menunjukkan bagaimana jenis kamar yang dipesan (misalnya, Room_Type 1, Room_Type 2) berhubungan dengan status pemesanan.")
    fig_room_type, ax_room_type = plt.subplots(figsize=(12, 6))
    sns.countplot(x='room_type_reserved', hue='booking_status', data=df, ax=ax_room_type)
    ax_room_type.set_title('Pengaruh Jenis Kamar yang Dipesan terhadap Status Pemesanan')
    ax_room_type.set_xlabel('Jenis Kamar')
    ax_room_type.set_ylabel('Jumlah Pemesanan')
    ax_room_type.legend(title='Status Pemesanan')
    st.pyplot(fig_room_type)
    st.write("Ini mengindikasikan bahwa preferensi jenis kamar dapat memengaruhi keputusan pembatalan. Hotel dapat menganalisis lebih lanjut mengapa jenis kamar tertentu memiliki tingkat pembatalan yang lebih tinggi dan menyesuaikan strategi pemasaran atau harga untuk mengurangi pembatalan.")

    # 7. Bagaimana pengaruh tipe segmen pasar terhadap status pemesanan?
    st.header('7. Bagaimana pengaruh tipe segmen pasar terhadap status pemesanan?')
    st.write("Visualisasi ini menunjukkan bagaimana tipe segmen pasar (misalnya, Online, Offline, Corporate) berhubungan dengan status pemesanan.")
    fig_market_segment, ax_market_segment = plt.subplots(figsize=(12, 6))
    sns.countplot(x='market_segment_type', hue='booking_status', data=df, ax=ax_market_segment)
    ax_market_segment.set_title('Pengaruh Tipe Segmen Pasar terhadap Status Pemesanan')
    ax_market_segment.set_xlabel('Tipe Segmen Pasar')
    ax_market_segment.set_ylabel('Jumlah Pemesanan')
    ax_market_segment.legend(title='Status Pemesanan')
    st.pyplot(fig_market_segment)
    st.write("Ini mengindikasikan bahwa sumber pemesanan dapat memengaruhi keputusan pembatalan. Hotel dapat menyesuaikan strategi pemasaran dan promosi berdasarkan segmen pasar untuk mengurangi pembatalan. Misalnya, menawarkan diskon khusus untuk pemesanan langsung (offline) atau meningkatkan pengalaman pengguna di platform online.")

    st.header('Feature Engineering - Outlier Handling')
    st.write("Visualisasi outlier sebelum dan sesudah capping")

    def cap_outliers(data, columns):
        for col in columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = data[col].clip(lower_bound, upper_bound)
        return data

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    X_train = df.copy()
    X_train = cap_outliers(X_train, numerical_cols)

    # Visualize outliers before and after capping (example for avg_price_per_room)
    fig_outlier, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df['avg_price_per_room'], ax=ax1)
    ax1.set_title('Before Outlier Handling')
    sns.boxplot(y=X_train['avg_price_per_room'], ax=ax2)
    ax2.set_title('After Outlier Handling')
    st.pyplot(fig_outlier)
    st.write("Dari visualisasi, terdapat banyak outlier yang dapat memengaruhi model machine learning. Outlier dapat menarik garis regresi yang berdampak pada prediksi yang bias atau tidak akurat. Maka dilakukan capping untuk penyesuaian. Setelah penanganan outlier, whisker boxplot menjadi lebih pendek, menunjukkan rentang nilai yang lebih kecil dan mengurangi pengaruh nilai-nilai ekstrem.")