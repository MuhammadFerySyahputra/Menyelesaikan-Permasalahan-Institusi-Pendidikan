

# Proyek Akhir: Menyelesaikan Permasalahan Edukasi Jaya Jaya Institut

  

## Business Understanding
#### 1. Pendahuluan
Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan dikenal luas atas kualitas lulusannya. Namun demikian, tantangan besar yang saat ini dihadapi adalah tingginya angka dropout banyak siswa yang tidak menyelesaikan studi mereka.

Dropout bukan hanya berdampak pada reputasi institusi, tetapi juga pada aspek finansial dan keberlangsungan pendidikan siswa. Oleh karena itu, pihak manajemen ingin melakukan deteksi dini terhadap siswa yang berpotensi dropout agar dapat diberikan intervensi dan bimbingan khusus sejak awal.

Untuk itu, pendekatan berbasis data science diperlukan guna mengidentifikasi siswa berisiko tinggi berdasarkan data historis performa akademik dan demografis.

#### 2. Masalah yang Dihadapi
Dropout mahasiswa memiliki dampak jangka panjang, baik bagi institusi maupun siswa:

- Menurunkan tingkat kelulusan

- Mempengaruhi akreditasi dan reputasi institusi

- Meningkatkan biaya administrasi akibat pengelolaan ulang sistem akademik

- Meningkatkan risiko kegagalan karier bagi siswa yang tidak menyelesaikan studi

Namun, mengidentifikasi siswa yang berpotensi dropout bukanlah hal mudah. Dibutuhkan sistem prediksi otomatis berbasis data agar tim akademik dapat mengambil keputusan yang tepat secara cepat.

#### 3. Pentingnya Penyelesaian Masalah dengan Pendekatan Data-Driven
Dengan pendekatan berbasis data, Jaya Jaya Institut dapat:

- Mengidentifikasi pola dan ciri khas siswa yang cenderung dropout

- Memprediksi risiko dropout sejak semester awal

- Menyediakan insight visual bagi dosen wali, akademik, dan manajemen kampus

- Memberikan intervensi yang lebih personal dan tepat sasaran

- Meningkatkan keberhasilan studi dan retensi siswa

  

### Permasalahan Bisnis

  
#### Problem Statements
- Bagaimana membangun model prediktif dropout mahasiswa berdasarkan data akademik dan demografis siswa?
- Algoritma klasifikasi mana yang paling optimal dalam mendeteksi risiko dropout?
- Bagaimana menyajikan hasil prediksi dan insight dalam bentuk dashboard interaktif agar mudah dipantau oleh pihak kampus?
  
#### Goals
- Membangun model klasifikasi untuk memprediksi apakah siswa akan dropout (1) atau tidak (0)
- Menggunakan model Random Forest Classifier untuk menangkap pola non-linear dari data siswa
- Mengevaluasi performa model dengan metrik: AUC ROC, Confusion Matrix, dan Classification Report
- Mengembangkan dashboard interaktif yang menyajikan performa model dan distribusi risiko dropout siswa

#### Solution Statements
- Solusi 1: Membangun model prediksi retensi menggunakan Random Forest Classifier yang dapat menangkap hubungan kompleks antar fitur 
- Solusi 2: Menggunakan evaluasi berbasis AUC untuk menilai kemampuan model dalam memisahkan mahasiswa yang dropout maupun yang tidak, serta validasi menggunakan Confusion Matrix dan Classification Report untuk melihat detail performa klasifikasi.
- Solusi 3: Mengembangkan dashboard interaktif yang menyajikan insight secara real-time dengan visualisasi yang mudah dipahami, 

  

### Cakupan Proyek
Cakupan proyek ini mencakup seluruh tahapan pemrosesan data hingga pembuatan aplikasi prediksi yang dapat digunakan oleh pihak kampus. Proyek dimulai dengan eksplorasi dan pra-pemrosesan data menggunakan dataset students_performance yang berisi informasi demografis dan performa akademik siswa. Dalam tahap ini dilakukan seleksi fitur penting seperti nilai semester, umur, dan biaya kuliah, serta penanganan missing value dan encoding untuk fitur kategorikal. Selanjutnya, dilakukan pembangunan model prediktif menggunakan algoritma Random Forest Classifier yang dievaluasi dengan metrik AUC, Confusion Matrix, dan Classification Report. Model yang telah dilatih disimpan dalam format .joblib agar dapat diintegrasikan ke dalam aplikasi. Aplikasi prediksi kemudian dibangun menggunakan Streamlit, yang memungkinkan pihak kampus mengunggah file CSV dan secara otomatis memperoleh prediksi siswa yang berisiko dropout. Terakhir, dibuat dashboard monitoring interaktif yang menampilkan visualisasi data siswa dan hasil klasifikasi, termasuk insight terkait siswa dengan risiko dropout tertinggi serta distribusi risiko berdasarkan umur, nilai, dan jurusan, sehingga mempermudah pihak akademik dalam pengambilan keputusan secara cepat dan tepat.

  

### Persiapan
---

#### Tentang Dataset
---

#### Deskripsi
Dataset ini berisi data performa akademik dan demografi siswa yang digunakan untuk prediksi risiko dropout di Jaya Jaya Institut. Data mencakup berbagai fitur penting seperti status pernikahan, mode penerimaan siswa, nilai rata-rata semester, status beasiswa, hutang, kehadiran, latar belakang pendidikan orang tua, serta beberapa variabel terkait lainnya yang berpotensi memengaruhi keputusan siswa untuk melanjutkan atau berhenti dari pendidikan. Label target pada dataset ini adalah status siswa, yang terdiri dari kategori seperti "Graduate" (lulus), "Dropout" (berhenti), dan "Enrolled" (masih aktif). Dataset ini dirancang untuk digunakan dalam eksplorasi data, pembuatan model prediktif klasifikasi, dan pengembangan alat monitoring performa siswa yang interaktif.

#### Konteks
Dataset ini dikumpulkan dengan tujuan membantu institusi pendidikan dalam memahami faktor-faktor yang berkontribusi terhadap tingginya angka dropout siswa. Nilai-nilai fitur mencerminkan kondisi nyata dan kompleksitas faktor-faktor sosial, akademik, dan ekonomi yang mempengaruhi keberlangsungan siswa dalam menempuh pendidikan. Dengan menganalisis dataset ini, institusi dapat mengidentifikasi pola risiko dropout secara lebih akurat, serta memanfaatkan hasil prediksi untuk melakukan intervensi bimbingan secara tepat waktu. Selain itu, dataset ini sangat berguna bagi para pengambil keputusan di bidang pendidikan dan pengelolaan institusi untuk mengembangkan dashboard visualisasi data yang membantu monitoring dan evaluasi performa belajar dan risiko dropout siswa secara real-time.

##### Sumber Dataset
Dataset ini diambil dari Github Dicoding dan dapat diakses melalui tautan berikut:  

https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

---


Setup environment:
# Panduan Menggunakan Aplikasi Prediksi Employee Attrition dengan Streamlit

## 1. Persiapan Awal
Sebelum menjalankan aplikasi, pastikan Anda sudah memiliki:

- **Python** (versi minimal 3.7) sudah terpasang di komputer Anda
- File aplikasi Streamlit (`main.py`)
- File model machine learning (`trained_pipeline.joblib`) di folder `model/`
- Dataset contoh (`dataset_fitur_terbaik.csv`) untuk validasi struktur kolom
- File `requirements.txt` yang berisi daftar dependency

### Untuk Windows:
Membuat dan Mengaktifkan Virtual Environment (venv)
```bash
python -m venv venv
```

Mengaktifkan virtual environment
```bash
venv\Scripts\activate
```

### Untuk macOS/Linux:
Membuat dan Mengaktifkan Virtual Environment (venv)
```bash
python3 -m venv venv
```

Mengaktifkan virtual environment
```bash
source venv/bin/activate
```
## 2. Instalasi Dependency

Update pip terlebih dahulu :
```bash
python -m pip install --upgrade pip
```

install semua dependency yang dibutuhkan:
```bash
pip install -r requirements.txt
```

Isi file `requirements.txt` minimal harus mencakup:
```
streamlit
pandas
numpy
joblib
scikit-learn
```

## 3. Menjalankan Aplikasi
Buka terminal dan arahkan ke folder proyek, lalu jalankan:

```bash
streamlit run main.py
```

Aplikasi akan otomatis terbuka di browser default Anda.

## 4. Menggunakan Aplikasi Prediksi

### a. Unggah File CSV
- Klik tombol "Browse files" untuk mengupload file CSV
- Pastikan file memiliki kolom yang sama dengan data training (kecuali kolom Status)
- Jika kolom tidak sesuai, akan muncul pesan error

### b. Preview Data
- Setelah upload berhasil, akan ditampilkan 5 baris pertama data
- Periksa apakah data sudah sesuai sebelum prediksi

### c. Proses Prediksi
- Klik tombol "Jalankan Prediksi Sekarang"
- Aplikasi akan menampilkan hasil prediksi berupa:
  - `Predicted_Probability`: Probabilitas Status (0-1)
  - `Predicted_Label`: Prediksi "Dropout" atau "Graduate"

### d. Download Hasil
- Klik tombol "Download Hasil Prediksi CSV" untuk menyimpan hasil
- File CSV akan berisi data asli + kolom prediksi

## 5. Troubleshooting
- Pastikan semua file berada di lokasi yang benar
- Periksa versi Python dan library yang terinstall
- Jika ada error, baca pesan yang muncul di terminal atau aplikasi

## Link Deployment ke Streamlit Cloud Community

Untuk mempermudah pihak institusi dalam melakukan prediksi dan pemantauan performa mahasiswa secara real-time, aplikasi ini telah dideploy secara publik menggunakan **Streamlit Cloud Community**. Melalui platform ini, pengguna dapat langsung mengakses aplikasi tanpa perlu instalasi tambahan.

🔗 **Akses aplikasi melalui tautan berikut:**  
[https://menyelesaikan-permasalahan-institusi-pendidikan-7wfxd6wtfjchvo.streamlit.app/](https://menyelesaikan-permasalahan-institusi-pendidikan-7wfxd6wtfjchvo.streamlit.app/)

Aplikasi ini memungkinkan pihak kampus untuk mengunggah file data mahasiswa dan langsung mendapatkan hasil prediksi risiko dropout, serta melihat dashboard monitoring performa mahasiswa secara interaktif.


## 📊 Business Dashboard
![Gambar Contoh](https://imghost.net/ib/3aIIRaRjw1IDCob_1750287177.png)

Dashboard ini dirancang untuk membantu pihak Jaya Jaya Institut dalam memantau performa akademik mahasiswa dan mendeteksi risiko dropout sejak dini. Dengan visualisasi yang interaktif dan informatif, dashboard ini memberikan wawasan menyeluruh yang mendukung pengambilan keputusan secara data-driven. Berikut adalah bagian-bagian utama yang ditampilkan:

- Jumlah Mahasiswa & Proporsi Status
Menunjukkan total populasi mahasiswa (3.630 mahasiswa) dan proporsi yang telah lulus (Graduate) versus yang mengalami dropout. Tampak bahwa 39,1% mahasiswa tidak menyelesaikan studi, sebuah angka yang perlu mendapat perhatian khusus.

- Distribusi Nilai Semester 1 & Semester 2
Visualisasi ini memperlihatkan bahwa mahasiswa dropout memiliki rerata nilai yang lebih rendah secara signifikan di kedua semester dibandingkan mahasiswa yang lulus. Hal ini menunjukkan bahwa penurunan performa akademik sejak awal merupakan indikator kuat risiko dropout.

- Hubungan Status dengan Perpindahan Mahasiswa
Data menunjukkan bahwa mahasiswa yang dropout memiliki kecenderungan lebih tinggi untuk dipindahkan, yang bisa menjadi sinyal ketidaksesuaian antara mahasiswa dan program studi awalnya.

- Hubungan Status dengan Beasiswa
Menariknya, proporsi mahasiswa dropout yang mendapat beasiswa cenderung lebih kecil dibanding yang lulus, mengindikasikan bahwa dukungan finansial mungkin berkontribusi terhadap keberhasilan studi.

- Hubungan Status dengan Pembayaran Kuliah
Visualisasi ini menegaskan bahwa mahasiswa yang memiliki tunggakan atau tidak membayar kuliah secara penuh memiliki risiko dropout yang jauh lebih tinggi, sehingga pembayaran kuliah menjadi salah satu indikator penting dalam deteksi dini dropout.

- Rekomendasi Action Items
Bagian ini menyajikan rekomendasi strategis yang bisa segera diimplementasikan institusi, mulai dari intervensi akademik bagi mahasiswa dengan nilai rendah, penyesuaian sistem pembayaran, dukungan sosial bagi mahasiswa displaced, hingga reformasi kebijakan kurikulum.

> https://lookerstudio.google.com/reporting/f247866a-d863-4c7e-a149-d7132683d059 

---

## 📌 Conclusion

Dashboard analitik ini berhasil mengungkap berbagai faktor signifikan yang berkontribusi terhadap risiko dropout mahasiswa di Jaya Jaya Institut. Ditemukan bahwa mahasiswa yang memiliki nilai rendah pada semester awal, tidak membayar kuliah, tidak mendapatkan beasiswa, dan mengalami perpindahan cenderung memiliki risiko dropout yang lebih tinggi. Dengan menggunakan data ini, institusi kini memiliki dasar yang kuat untuk melakukan intervensi dini secara lebih efektif.

---

## 🎯 Rekomendasi Action Items

Berikut adalah beberapa langkah strategis berbasis temuan data yang dapat diterapkan:

### 1. Fokus Penanganan Mahasiswa dengan Nilai Semester 1 Rendah
- **Insight**: Visualisasi menunjukkan adanya penurunan nilai pada mahasiswa yang berpotensi dropout.
- **Tindakan**: Kembangkan program bimbingan akademik atau remedial khusus untuk mahasiswa dengan nilai semester 1 rendah.

### 2. Penurunan Status Pembayaran Kuliah
- **Insight**: Mahasiswa yang tidak membayar kuliah memiliki risiko dropout yang sangat tinggi.
- **Tindakan**: Buat sistem pemantauan pembayaran dan sediakan skema bantuan finansial bagi mahasiswa dengan risiko tinggi.

### 3. Pendampingan Mahasiswa dengan Status Displaced
- **Insight**: Mahasiswa yang berpindah program studi atau lokasi memiliki kecenderungan lebih tinggi untuk dropout.
- **Tindakan**: Adakan bimbingan sosial dan akademik secara khusus bagi mahasiswa ini.

### 4. Manfaatkan Skema Beasiswa sebagai Pilar Motivasi
- **Insight**: Statistik menunjukkan hubungan kuat antara beasiswa dan kelulusan.
- **Tindakan**: Tingkatkan alokasi beasiswa untuk mahasiswa berisiko sebagai insentif belajar dan retensi studi.

### 5. Audit dan Perbaikan Kurikulum
- **Insight**: Data menunjukkan korelasi kuat antara performa awal dan dropout.
- **Tindakan**: Evaluasi desain mata kuliah awal semester, dan sesuaikan metode pengajaran agar lebih adaptif terhadap kebutuhan mahasiswa.



