import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0) Load data
URL = "diabetes.csv"
df = pd.read_csv(URL)

# 4A) Pilih kolom numerik
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Kolom numerik:", list(num_cols))

# 4B) Min-Max Scaling manual (tanpa library eksternal)
df_norm = df.copy()
for c in num_cols:
    col_min = df_norm[c].min()
    col_max = df_norm[c].max()
    if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
        # Skip jika kolom kosong atau konstan
        continue
    df_norm[c + "_norm"] = (df_norm[c] - col_min) / (col_max - col_min)

# 4C) Tampilkan 5 baris teratas setelah normalisasi
print("\nLima baris teratas setelah normalisasi:")
print(df_norm.head())


# 4D) (Opsional) Plot histogram kolom numerik terpilih sebelum & sesudah normalisasi
kolom_contoh = num_cols[0] if len(num_cols) > 0 else None
if kolom_contoh and (kolom_contoh + "_norm") in df_norm.columns:
    plt.figure()
    plt.hist(df[kolom_contoh].dropna(), bins=30)
    plt.title(f"Distribusi Asli: {kolom_contoh}")
    plt.xlabel(kolom_contoh)
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.hist(df_norm[kolom_contoh + "_norm"].dropna(), bins=30)
    plt.title(f"Distribusi Setelah Normalisasi: {kolom_contoh}_norm")
    plt.xlabel(kolom_contoh + "_norm")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.show()
