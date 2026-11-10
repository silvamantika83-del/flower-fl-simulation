# 🌸 Flower Federated Learning - SuperGrid Simulation Dashboard

Dashboard interaktif berbasis web untuk simulasi **Federated Learning** menggunakan arsitektur **Flower SuperGrid v1.40** (1 Server SuperLink & 3 Client SuperNodes) dan algoritma **FedAvg (Federated Averaging)**.

Aplikasi ini siap di-deploy secara langsung ke **Streamlit Community Cloud (`share.streamlit.io`)**.

---

## 🚀 Fitur Unggulan

1. **Topologi Jaringan Visual 1 Server, 3 Client:**
   - 1 Server Coordinator (Flower SuperLink) menjalankan algoritma FedAvg.
   - 3 Client Workers (SuperNodes: Alpha [100 sampel], Beta [150 sampel], Gamma [80 sampel]).
2. **Simulasi Siklus Lengkap Federated Learning:**
   - Parameter Broadcast ($W_t$)
   - Local Training pada dataset privat masing-masing SuperNode
   - Upload parameter & gradien bobot
   - Agregasi bobot terbobot dengan algoritma FedAvg
3. **Simulasi Kegagalan Jaringan (Node Dropout):**
   - Menguji keandalan sistem jika SuperNode 3 (Gamma) mengalami network drop/unreachable pada ronde ke-3 ke atas.
4. **Grafik Metrik Real-Time (Plotly):**
   - Kurva Akurasi Global & Per Klien (0 - 100%).
   - Kurva Penurunan Loss Model.
   - Tabel riwayat metrik per ronde.
5. **Live Event Logging:**
   - Menampilkan rekaman log sinkronisasi gRPC dan event SuperLink secara real-time.
6. **Dokumentasi Teori & Arsitektur:**
   - Penjelasan teknis Flower SuperGrid dan formulasi matematis FedAvg:
     $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k$$

---

## 🛠️ Cara Deploy ke Streamlit Cloud (streamlit.io)

### Langkah 1: Push ke GitHub
1. Buat repository baru di GitHub Anda (misalnya: `flower-fl-streamlit`).
2. Masukkan semua file dari folder ini (`streamlit_app.py`, `requirements.txt`, `.streamlit/config.toml`, dan `README.md`).
3. Commit dan push ke repository GitHub Anda.

### Langkah 2: Deploy di share.streamlit.io
1. Kunjungi **[share.streamlit.io](https://share.streamlit.io)** dan masuk (sign-in) dengan akun GitHub Anda.
2. Klik tombol **"Create app"** / **"New app"**.
3. Pilih repository Anda: `<username_github>/flower-fl-streamlit`.
4. Pilih Branch: `main` (atau `master`).
5. Main file path: `streamlit_app.py`.
6. Klik **"Deploy!"**.

Aplikasi Anda akan live dalam waktu 1-2 menit dengan URL publik resmi:
`https://<nama-app>.streamlit.app` yang siap dikumpulkan sebagai tugas!

---

## 💻 Menjalankan Secara Lokal (Opsional)

Jika ingin mencoba di komputer lokal sebelum deploy:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.
