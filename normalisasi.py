# =========================================
# Analisis Missing, Imputasi, Normalisasi
# Dataset: diabetes.csv (Pima Indians Diabetes)
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 0) LOAD DATA
URL = "diabetes.csv"  # ganti path jika perlu
df = pd.read_csv(URL)
print("Shape awal:", df.shape)
print("Lima baris teratas:")
print(df.head())
print("\nInfo awal:")
print(df.info())

# -------------------------------------------------------
# 1) TANDAI '0' SEBAGAI MISSING (NaN) PADA KOLOM FISIOLOGIS
#    NOTE:
#    - Biasanya 0 pada kolom ini dianggap hilang (tidak realistis):
#      Glucose, BloodPressure, SkinThickness, Insulin, BMI
#    - Jangan tandai 0 sebagai NaN untuk Pregnancies & Outcome, karena 0 valid.
# -------------------------------------------------------
cols_must_not_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df_fixed = df.copy()
df_fixed[cols_must_not_zero] = df_fixed[cols_must_not_zero].replace(0, np.nan)

# 2) RINGKASAN & GRAFIK MISSING
missing_counts = df_fixed.isna().sum().sort_values(ascending=False)
missing_pct = (df_fixed.isna().mean() * 100).round(2).sort_values(ascending=False)

summary_missing = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_pct": missing_pct
})
print("\nRingkasan missing per kolom:")
print(summary_missing)

# --- Grafik: jumlah missing per kolom ---
plt.figure(figsize=(10, 5))
missing_counts.plot(kind='bar')
plt.title('Jumlah Missing per Kolom')
plt.xlabel('Kolom')
plt.ylabel('Jumlah Missing')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Grafik: persentase missing per kolom ---
plt.figure(figsize=(10, 5))
missing_pct.plot(kind='bar')
plt.title('Persentase Missing per Kolom (%)')
plt.xlabel('Kolom')
plt.ylabel('Persentase (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Peta missing (baris x kolom) ---
plt.figure(figsize=(10, 6))
plt.imshow(df_fixed.isna(), aspect='auto', interpolation='nearest')
plt.title('Peta Missing per Baris/Kolom')
plt.xlabel('Kolom')
plt.ylabel('Baris')
plt.yticks([])
plt.xticks(
    ticks=np.arange(len(df_fixed.columns)),
    labels=df_fixed.columns, rotation=45, ha='right'
)
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 3) IMPUTASI NILAI HILANG
#    - Imputasi median untuk kolom numerik (kecuali label/target 'Outcome')
# -------------------------------------------------------
df_impute = df_fixed.copy()

# kolom numerik
num_cols = df_impute.select_dtypes(include=[np.number]).columns.tolist()
# fitur (tanpa label)
feature_cols = [c for c in num_cols if c.lower() != "outcome"]

# imputasi median pada fitur
for c in feature_cols:
    med = df_impute[c].median()
    df_impute[c] = df_impute[c].fillna(med)

print("\nTotal missing pada fitur setelah imputasi:",
      int(df_impute[feature_cols].isna().sum().sum()))

# -------------------------------------------------------
# 4) NORMALISASI MIN–MAX [0,1] (hanya fitur)
# -------------------------------------------------------
df_norm = df_impute.copy()
for c in feature_cols:
    cmin, cmax = df_norm[c].min(), df_norm[c].max()
    if pd.notna(cmin) and pd.notna(cmax) and cmax > cmin:
        df_norm[c] = (df_norm[c] - cmin) / (cmax - cmin)
    else:
        df_norm[c] = 0.0  # jika konstanta atau kosong

print("\nLima baris teratas setelah normalisasi:")
print(df_norm.head())

# -------------------------------------------------------
# 5) GRAFIK DISTRIBUSI & BOX PLOT SETELAH NORMALISASI
# -------------------------------------------------------

# (a) Histogram gabungan (grid) untuk semua fitur ternormalisasi
import math
k = len(feature_cols)
ncols = 3
nrows = math.ceil(k / ncols)

plt.figure(figsize=(ncols*5, nrows*3.2))
for i, c in enumerate(feature_cols, 1):
    plt.subplot(nrows, ncols, i)
    plt.hist(df_norm[c].dropna(), bins=30)
    plt.title(c)
    plt.xlabel("Nilai (0–1)")
    plt.ylabel("Frekuensi")
plt.tight_layout()
plt.show()

# (b) Boxplot semua fitur ternormalisasi
plt.figure(figsize=(max(10, len(feature_cols)*0.9), 5))
plt.boxplot([df_norm[c].dropna() for c in feature_cols], labels=feature_cols, vert=True)
plt.title("Boxplot Fitur Ternormalisasi (0–1)")
plt.ylabel("Skala 0–1")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# 6) SIMPAN HASIL
# -------------------------------------------------------
df_fixed.to_csv("diabetes_zero_as_missing.csv", index=False)
df_impute.to_csv("diabetes_imputed.csv", index=False)
df_norm.to_csv("diabetes_normalized.csv", index=False)
print("\nFile tersimpan:")
print("- diabetes_zero_as_missing.csv")
print("- diabetes_imputed.csv")
print("- diabetes_normalized.csv")
