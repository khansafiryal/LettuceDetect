# Reproduksi Paper LettuceDetect

Panduan ini menjelaskan cara mereproduksi hasil utama paper
**LettuceDetect: A Hallucination Detection Framework for RAG Applications**
([arXiv 2502.17125](https://arxiv.org/abs/2502.17125)) di dataset RAGTruth,
dari unduh data sampai angka evaluasi.

## Ringkasan

Model di paper adalah *token classifier* berbasis ModernBERT. Model menerima
konteks, pertanyaan, dan jawaban, lalu menandai setiap token jawaban sebagai
didukung konteks (0) atau halusinasi (1).

Tidak ada satu file "main". Pipeline terdiri dari empat tahap yang dijalankan
berurutan, dan output tiap tahap menjadi input tahap berikutnya:

| Tahap | File | Input | Output |
|---|---|---|---|
| 1. Download data | (unduh dari GitHub RAGTruth) | – | `data/ragtruth/response.jsonl`, `source_info.jsonl` |
| 2. Preprocessing | `lettucedetect/preprocess/preprocess_ragtruth.py` | dua file `.jsonl` | `data/ragtruth/ragtruth_data.json` |
| 3. Training | `scripts/train.py` | `ragtruth_data.json` (split `train`) | `output/lettucedetect-<model>/` |
| 4. Evaluasi | `scripts/evaluate.py` | model + `ragtruth_data.json` (split `test`) | Precision / Recall / F1 |

Keempat tahap bisa dijalankan sekaligus dengan `reproduce.ps1` (lihat bagian
[Cara cepat](#cara-cepat-satu-perintah)).

## Prasyarat

- Python 3.10+ (diuji dengan 3.12.10)
- GPU NVIDIA dengan CUDA (diuji dengan RTX 5050 Laptop, 8 GB)
- PyTorch dengan CUDA (diuji dengan torch 2.7.0+cu128, transformers 5.20.2)

Instal package dari root repo **dalam mode editable** (`-e`):

```powershell
pip install -e .
```

> **Penting:** jika `lettucedetect` pernah diinstal dengan `pip install lettucedetect`
> (bukan `-e .`), skrip di `scripts/` akan memakai salinan di `site-packages`
> dan mengabaikan kode di repo ini, termasuk opsi `--bf16`/`--grad-accum`.
> Cek dengan `python -c "import lettucedetect; print(lettucedetect.__file__)"`
> dari folder lain. Path-nya harus mengarah ke folder repo.

Cek bahwa GPU terdeteksi:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Windows:** setel `$env:PYTHONUTF8=1` sebelum menjalankan skrip Python
> agar file data dibaca sebagai UTF-8. Tanpa ini, preprocessing bisa gagal
> dengan `UnicodeDecodeError`.

## Cara cepat (satu perintah)

Dari root repo di terminal VS Code (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File reproduce.ps1
```

Opsi yang tersedia:

| Opsi | Kegunaan |
|---|---|
| `-Model large` | Pakai ModernBERT-large (default `base`) |
| `-BatchSize`, `-GradAccum` | Ukuran batch per langkah dan jumlah akumulasi gradien (default 2 dan 4, jadi batch efektif 8) |
| `-Bf16:$false` | Matikan bf16 (default menyala). Setting asli paper: `-BatchSize 8 -GradAccum 1 -Bf16:$false`, butuh GPU besar |
| `-SkipDownload` | Lewati tahap 1 jika data sudah ada |
| `-SkipPreprocess` | Lewati tahap 2 jika `ragtruth_data.json` sudah ada |
| `-SkipTraining` | Hanya evaluasi model yang sudah ada di `output/lettucedetect-<model>/` |

Setiap kali dijalankan, skrip membuat folder log baru
`output/logs/<model>_<tanggal_jam>/` berisi:

| File log | Isi |
|---|---|
| `00_environment.log` | Commit git, versi Python/torch/transformers, nama GPU, hyperparameter |
| `01_download.log` | Ukuran file dataset yang diunduh |
| `02_preprocess.log` | Jumlah sampel per split |
| `03_train.log` | Loss dan F1 dev per epoch, checkpoint terbaik |
| `04_eval_example_level.log` | Hasil evaluasi example-level |
| `04_eval_span_level.log` | Hasil evaluasi span-level |

Setiap log mencatat waktu mulai, waktu selesai, dan durasi. Folder log inilah
bukti/dokumentasi dari sebuah run.

## Langkah manual (tahap demi tahap)

Jika ingin menjalankan dan memahami setiap tahap secara terpisah:

### Tahap 1 — Download RAGTruth

```powershell
$env:PYTHONUTF8=1
mkdir data/ragtruth -Force
curl.exe -L -o data/ragtruth/response.jsonl https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl
curl.exe -L -o data/ragtruth/source_info.jsonl https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl
```

- `response.jsonl`: jawaban yang dihasilkan LLM beserta anotasi span halusinasi (posisi karakter `start`/`end`) dan split (`train`/`test`).
- `source_info.jsonl`: prompt/konteks sumber untuk tiap jawaban, dengan tipe tugas QA, Summary, atau Data2txt.

Ukuran yang diharapkan: `response.jsonl` sebesar 21.458.735 byte dan `source_info.jsonl` sebesar 15.117.971 byte.

### Tahap 2 — Preprocessing

```powershell
python lettucedetect/preprocess/preprocess_ragtruth.py --input_dir data/ragtruth --output_dir data/ragtruth
```

Skrip ini menggabungkan setiap jawaban dengan konteks sumbernya dan menyimpan
semuanya dalam satu format seragam (`HallucinationSample`) di
`data/ragtruth/ragtruth_data.json`.

Hasil yang diharapkan: **17.790 sampel**, terdiri dari **15.090 train** dan **2.700 test**.

### Tahap 3 — Training

Untuk GPU 8 GB (setting yang dipakai di reproduksi ini):

```powershell
python scripts/train.py `
    --ragtruth-path data/ragtruth/ragtruth_data.json `
    --model-name answerdotai/ModernBERT-base `
    --output-dir output/lettucedetect-base `
    --batch-size 2 --grad-accum 4 --bf16 --epochs 6 --learning-rate 1e-5
```

Setting asli paper (butuh GPU besar seperti A100): `--batch-size 8` tanpa `--grad-accum` dan tanpa `--bf16`.

Yang terjadi di dalam `train.py`:

1. Membaca `ragtruth_data.json` dan mengambil sampel split `train`.
2. Menyisihkan 10% sampel train secara acak sebagai *dev set* (seed tetap).
3. Tokenisasi `konteks + pertanyaan + jawaban`. Hanya token jawaban yang diberi label; token prompt diabaikan (`-100`).
4. Fine-tuning ModernBERT sebagai token classifier dengan 2 label.
5. Setelah tiap epoch, menghitung F1 kelas halusinasi di dev set, lalu **menyimpan model terbaik** ke `--output-dir`.

Hyperparameter paper: 6 epoch, batch size 8, learning rate 1e-5, satu GPU A100.
Untuk model large, ganti `--model-name` menjadi `answerdotai/ModernBERT-large`.
Model large kemungkinan tidak muat di GPU 8 GB.

#### Perbedaan dari paper (GPU 8 GB)

Setting paper (batch 8, fp32, input hingga 4.096 token) membutuhkan memori lebih
dari 8 GB. Di RTX 5050 Laptop, run tersebut gagal dengan `CUDA out of memory`
setelah memori luber ke RAM sistem (54 detik per langkah, sekitar 150 jam untuk
6 epoch). Hasil pengukuran singkat di GPU yang sama:

| Konfigurasi | Detik/langkah | Memori puncak | Estimasi 6 epoch |
|---|---|---|---|
| batch 8, fp32 (paper) | 54 | > 8 GB, crash | tidak selesai |
| batch 4, bf16 | 4,6 | 7,5 GB (luber) | ~26 jam |
| batch 2, fp32 | 2,7 | 6,6 GB | ~31 jam |
| **batch 2, bf16** | **0,35** | **4,7 GB** | **~3–4 jam** |

Karena itu, dua opsi ditambahkan ke `scripts/train.py` dan `lettucedetect/models/trainer.py`.
Nilai default-nya mempertahankan perilaku asli:

- `--grad-accum 4`: gradien dari 4 batch kecil (masing-masing 2 sampel) dijumlahkan sebelum satu langkah optimizer, sehingga **batch efektif tetap 8** seperti di paper.
- `--bf16`: forward pass dijalankan dengan autocast bfloat16. Ini **satu-satunya perbedaan** dari paper. Pengaruhnya pada hasil biasanya kecil, tetapi bisa menggeser F1 beberapa persepuluh poin.

Catatan teknis: loss ModernBERT dirata-rata per token di dalam tiap batch kecil,
sehingga akumulasi 4×2 tidak identik secara matematis dengan satu batch berisi 8.
Perbedaannya kecil.

### Tahap 4 — Evaluasi

```powershell
# Apakah jawaban mengandung halusinasi? (metrik utama paper)
python scripts/evaluate.py --model_path output/lettucedetect-base --data_path data/ragtruth/ragtruth_data.json --evaluation_type example_level

# Seberapa tepat span halusinasi yang ditandai?
python scripts/evaluate.py --model_path output/lettucedetect-base --data_path data/ragtruth/ragtruth_data.json --evaluation_type span_level
```

Hasil ditampilkan per tipe tugas (Summary, Data2txt, QA) dan untuk seluruh
test set (`whole dataset`). Angka yang dibandingkan dengan paper adalah
**F1 kelas Hallucinated** pada baris `whole dataset`.

## Hasil acuan

### Angka di paper (example-level F1, test set RAGTruth)

| Model | F1 |
|---|---|
| lettucedetect-base (ModernBERT-base, 149M) | 76.8% |
| lettucedetect-large (ModernBERT-large, 395M) | 79.2% |

### Verifikasi pipeline dengan model resmi

Sebelum training, pipeline data dan evaluasi diverifikasi memakai model resmi
yang sudah dipublikasikan:

```powershell
python scripts/evaluate.py --model_path KRLabsOrg/lettucedect-base-modernbert-en-v1 --data_path data/ragtruth/ragtruth_data.json --evaluation_type example_level
```

Hasil di mesin uji (RTX 5050, transformers 5.20.2), kelas Hallucinated:

| Tipe tugas | Precision | Recall | F1 |
|---|---|---|---|
| QA | 60.6% | 71.3% | 65.5% |
| Summary | 53.9% | 47.6% | 50.5% |
| Data2txt | 89.3% | 86.5% | 87.9% |
| **Keseluruhan** | **76.6%** | **75.5%** | **76.1%** |

Selisihnya sekitar 0.7 poin dari angka paper (76.8%), kemungkinan karena
perbedaan versi library. Ini menunjukkan data dan skrip evaluasi sudah benar,
sehingga hasil training sendiri bisa dibandingkan dengan tabel ini.

### Hasil training sendiri

_Isi setelah menjalankan training (salin dari `output/logs/.../04_eval_example_level.log`)._

| Model | Batch × akumulasi | Presisi | Durasi training | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| base | 2 × 4 | bf16 | | | | |

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `UnicodeDecodeError` saat preprocessing | Jalankan `$env:PYTHONUTF8=1` dulu |
| `CUDA out of memory`, atau training sangat lambat (puluhan detik per langkah) | Pakai `--batch-size 2 --grad-accum 4 --bf16` (default `reproduce.ps1`) |
| `unexpected keyword argument 'grad_accum_steps'` | Package belum diinstal editable: jalankan `pip install -e .` |
| `running scripts is disabled on this system` | Jalankan dengan `powershell -ExecutionPolicy Bypass -File reproduce.ps1` |
| `torch.cuda.is_available()` bernilai `False` | Instal ulang PyTorch versi CUDA dari https://pytorch.org |
| `... is not a local folder and is not a valid model identifier` saat evaluasi | Training belum selesai atau `--model_path` salah |
| Hasil sedikit berbeda dari paper | Wajar (±1 poin): perbedaan GPU, batch size, versi library, dan nondeterminisme CUDA |
