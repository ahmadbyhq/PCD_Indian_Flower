# Klasifikasi Citra Bunga: Pengaruh Preprocessing Min-Max dan CLAHE

Proyek Final (UAS) mata kuliah **Pengolahan Citra Digital**.

## Deskripsi

Proyek ini menganalisis pengaruh preprocessing terhadap akurasi klasifikasi citra bunga, sekaligus membandingkan dua model CNN pada data yang sama.

- **Dataset:** [Indian Flower Images](https://www.kaggle.com/datasets/arjun2003/indian-flower-images/data) (Kaggle), 6.538 citra, 27 kelas, 224 x 224 RGB
- **Preprocessing:** Original (baseline), Min-Max Scaling, CLAHE
- **Model:** MobileNetV2, EfficientNetB0
- **Metrik:** accuracy, precision, recall, F1-score, confusion matrix

Setiap model dilatih dengan setiap metode preprocessing, sehingga ada 6 skenario:

| Skenario | Model | Preprocessing |
|:---:|---|---|
| A | MobileNetV2 | Original |
| B | MobileNetV2 | Min-Max |
| C | MobileNetV2 | CLAHE |
| D | EfficientNetB0 | Original |
| E | EfficientNetB0 | Min-Max |
| F | EfficientNetB0 | CLAHE |

## Alur Kerja

1. **Split** dataset satu kali menjadi train / val / test (70 / 15 / 15, stratified, seed tetap). Split yang sama dipakai untuk keenam skenario.
2. **Preprocessing** citra dengan Min-Max dan CLAHE, lalu simpan ke disk. Citra original dibaca langsung dari `data/raw/`.
3. **Training** setiap skenario dengan hyperparameter yang sama. Hanya model dan preprocessing yang berbeda.
4. **Evaluasi** di data test, lalu bandingkan hasilnya.

## Setup

Membutuhkan **Python 3.11**

```bash
git clone https://github.com/ahmadbyhq/PCD_Indian_Flower.git
cd PCD_Indian_Flower
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Unduh dataset dari Kaggle dan letakkan di `data/raw/`, satu folder untuk setiap kelas:

```
data/raw/
├── champaka/
│   ├── champaka1.jpg
│   └── ...
└── ... (27 folder kelas)
```

## Cara Menjalankan

```bash
python src/data_split.py        # membuat file CSV train/val/test
python src/save_processed.py    # membuat citra hasil Min-Max dan CLAHE
```

Lanjut run notebooks (exploration_data dan prepocessing.ipynb)

Training & Evaluasi:

```bash
python src/train.py --model mobilenetv2 --prep original
python src/train.py --model mobilenetv2 --prep minmax
python src/train.py --model mobilenetv2 --prep clahe

python src/train.py --model efficientnetb0 --prep original
python src/train.py --model efficientnetb0 --prep minmax
python src/train.py --model efficientnetb0 --prep clahe

python src/evaluate.py --model mobilenetv2 --prep original
python src/evaluate.py --model mobilenetv2 --prep minmax
python src/evaluate.py --model mobilenetv2 --prep clahe

python src/evaluate.py --model efficientnetb0 --prep original
python src/evaluate.py --model efficientnetb0 --prep minmax
python src/evaluate.py --model efficientnetb0 --prep clahe
```

- `--model`: `mobilenetv2` atau `efficientnetb0`
- `--prep`: `original`, `minmax`, atau `clahe`

Jalankan sekali untuk tiap skenario dengan kombinasi yang berbeda.

Lanjut run results_analysis.ipynb untuk meilhat hasil model dengan semua skenario

## Struktur Project

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml            # pengaturan bersama (epoch, learning rate, dll.)
├── data/
│   ├── raw/                   # dataset asli (tidak di-commit)
│   ├── processed/             # citra Min-Max dan CLAHE dalam format PNG (tidak di-commit)
│   │   ├── minmax/
│   │   └── clahe/
│   └── splits/                # train.csv, val.csv, test.csv
├── notebooks/
│   ├── exploration_data.ipynb        # gambaran dataset dan distribusi kelas
│   ├── preprocessing_demo.ipynb   # perbandingan before/after dan histogram
│   └── results_analysis.ipynb     # analisis hasil dan confusion matrix
├── src/
│   ├── data_split.py          # membuat stratified split train/val/test
│   ├── preprocessing.py       # fungsi original, Min-Max, dan CLAHE
│   ├── save_processed.py      # menyimpan citra hasil preprocessing ke data/processed/
│   ├── dataset.py             # memuat citra dari file CSV split
│   ├── models.py              # membangun MobileNetV2 dan EfficientNetB0
│   ├── train.py               # training satu skenario
│   ├── evaluate.py            # menghitung metrik dan confusion matrix
│   └── utils.py               # fungsi bantu
├── results/
│   ├── metrics/               # metrik tiap skenario
│   ├── confusion_matrix/
│   └── figures/               # gambar untuk laporan
└── models/                    # bobot model hasil training (tidak di-commit)
```

File yang belum dibuat (`dataset.py`, `models.py`, `train.py`, `evaluate.py`) akan ditambahkan seiring berjalannya project.

## Metode Preprocessing

- **Original:** nilai piksel dibagi 255 (rentang 0 sampai 1). Dipakai sebagai baseline.
- **Min-Max:** setiap citra diskalakan dengan nilai min dan max-nya sendiri: `(x - min) / (max - min)`.
- **CLAHE:** peningkatan kontras pada channel L di ruang warna LAB, supaya warna bunga tidak bergeser. Parameter: `clipLimit=2.0`, `tileGridSize=(8, 8)`.

## Teknologi

Python 3.11.9, TensorFlow 2.21.0, OpenCV, scikit-learn, pandas, NumPy, Matplotlib, Seaborn.