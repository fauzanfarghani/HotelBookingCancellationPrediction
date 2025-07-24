# Prediksi Pembatalan Reservasi Hotel

## Repository Outline
Berikut adalah isi dari file yang ada di repository ini:

1. description.md - Penjelasan gambaran umum proyek.
2. P1M2_fauzan_farghani_inf.ipynb - Notebook yang berisi proses pengembangan model machine learning untuk prediksi pembatalan reservasi hotel, termasuk eksplorasi data, feature engineering, pemodelan, evaluasi, dan hyperparameter tuning.
3. P1M2_fauzan_farghani_inf.ipynb - Notebook untuk inference, digunakan untuk menguji prediksi pembatalan reservasi hotel menggunakan model Random Forest yang telah dituning.
4. rf_tuned_pipeline.pkl - File model Random Forest yang telah dituning dan disimpan untuk keperluan prediksi.
5. Hotel_Reservations.csv - Dataset yang digunakan untuk Project ini.
6. P1M2_fauzan_farghani_conceptual.txt - informasi umum terkait cara kerja Cross Validation, Random Forest Algorithm dan bagging.
7. url.txt - URL Dataset dan URL deployment
8. file src - source code yang digunakan untuk deployment, berisikan app.py, eda.py, prediction.py, csv dataset dan model yang telah didefinisikan di nomor 4 dan 5.
9. app.py - main program untuk menjalankan website yang telah dideploy
10. eda.py - program untuk menjalankan sub-page dari website terkait exploratory data analysis
11. prediction.py - program untuk menjalankan model dengan menginput untuk memprediksi target output.

## Problem Background
Proyek ini bertujuan untuk memprediksi apakah reservasi hotel akan dibatalkan berdasarkan data reservasi (`Hotel_Reservations.csv`). Pembatalan reservasi dapat menyebabkan kerugian pendapatan dan masalah operasional bagi hotel. Dengan memprediksi pembatalan, hotel dapat mengelola inventaris kamar, mengoptimalkan strategi overbooking, merencanakan sumber daya operasional, dan menargetkan tamu berisiko tinggi dengan promosi khusus untuk mengurangi tingkat pembatalan.

## Project Output
Output proyek ini adalah:
- Model machine learning berbasis **Random Forest** yang telah dituning untuk memprediksi status pembatalan reservasi hotel (Canceled atau Not_Canceled).
- Pipeline model yang disimpan (`rf_tuned_pipeline.pkl`) untuk digunakan dalam prediksi pada data baru.
- Hasil evaluasi model dengan metrik **ROC-AUC** dan **F1-Score**, serta visualisasi seperti matriks konfusi dan kurva ROC.

## Data
Data yang digunakan adalah dataset `Hotel_Reservations.csv` yang berisi informasi reservasi hotel dengan kolom-kolom berikut:
- **Jumlah kolom**: 19 (termasuk fitur seperti `no_of_adults`, `no_of_children`, `lead_time`, `avg_price_per_room`, `no_of_special_requests`, dll., dan target `booking_status`).
- **Jumlah baris**: 36275 baris.
- **Karakteristik data**:
  - Fitur numerik (contoh: `lead_time`, `avg_price_per_room`) dan kategorik (contoh: `type_of_meal_plan`, `market_segment_type`).
  - Tidak ada informasi missing values dalam notebook, tetapi tetap dilakukan pemeriksaan nilai hilang.
  - Target: `booking_status` (Not_Canceled, Canceled).
- **Sumber data**: Tidak disebutkan secara spesifik, tetapi diasumsikan berasal dari dataset publik atau internal.

## Method
Metode yang digunakan adalah **supervised learning** dengan pendekatan klasifikasi. Model utama yang dipilih adalah **Random Forest** setelah membandingkan performa dengan model lain seperti KNN, SVC, Decision Tree, dan XGBoost. Proses pengembangan meliputi:
- **Eksplorasi Data (EDA)**: Analisis distribusi fitur, kardinalitas, dan outlier.
- **Feature Engineering**: Pemilihan fitur berdasarkan feature importance, penanganan kardinalitas, dan splitting data.
- **Pemodelan**: Implementasi Random Forest dengan pipeline yang mencakup preprocessing (OneHotEncoder untuk fitur kategorik, StandardScaler untuk fitur numerik).
- **Hyperparameter Tuning**: Menggunakan GridSearchCV untuk mengoptimalkan parameter Random Forest (n_estimators, max_depth, min_samples_split, min_samples_leaf).
- **Evaluasi**: Menggunakan metrik ROC-AUC, F1-Score, dan confusion matrix untuk mengevaluasi performa model.

## Stacks
- **Bahasa Pemrograman**: Python, streamlit
- **Tools**: Jupyter Notebook, VSCode, HuggingFace
- **Library Python**:
  - `pandas`, `numpy`: Pengolahan data.
  - `matplotlib`, `seaborn`: Visualisasi data.
  - `scikit-learn`: Preprocessing (StandardScaler, OneHotEncoder), pemodelan (KNeighborsClassifier, SVC, DecisionTreeClassifier, RandomForestClassifier), evaluasi (classification_report, confusion_matrix, roc_curve, auc), dan hyperparameter tuning (GridSearchCV).
  - `xgboost`: Model XGBoost untuk perbandingan.
  - `joblib`: Penyimpanan model.

## Reference
- [Dataset Hotel Reservations](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) (diasumsikan, karena tidak disebutkan eksplisit).
- [Basic Writing and Syntax on Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Contoh README](https://github.com/fahmimnalfrzki/Swift-XRT-Automation)
- [Another Example](https://github.com/sanggusti/final_bangkit)
- [Additional Reference](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)