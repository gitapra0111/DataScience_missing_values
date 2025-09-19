# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # 0) Load data
# URL = "diabetes.csv"
# df = pd.read_csv(URL)

# # 2A) Jumlah missing per kolom
# missing_count = df.isna().sum()
# print("Jumlah missing per kolom:")
# print(missing_count)

# # 2B) Persentase missing per kolom
# missing_pct = df.isna().mean().round(4) * 100
# print("\nPersentase missing per kolom (%):")
# print(missing_pct)

# # 2C) Contoh baris yang punya missing value (tampilkan 5 baris)
# print("\nContoh baris dengan missing value:")
# print(df[df.isna().any(axis=1)].head())

# # 2D) (Opsional) Visual sederhana jumlah missing per kolom
# ax = missing_count.plot(kind='bar', title='Jumlah Missing per Kolom')
# ax.set_xlabel('Kolom')
# ax.set_ylabel('Jumlah Missing')
# plt.tight_layout()
# plt.show()


# atau bisa menggunakan 

# import pandas as pd
# dataset = pd.read_csv("diabetes.csv")
# print("Cek missing value untuk setiap feature")
# print(dataset.isnull().sum())
# print("Hitung jumlah missing value")
# print("dataset.isnull().sum().sum()")



# fix code untuk missing value pada diabetes.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 0) Load data
URL = "diabetes.csv"
# URL = "DS_2.csv"
df = pd.read_csv(URL)

cols_must_not_zero = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
# cols_must_not_zero = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"] nilai yang seharuhnya tidak boleh 0
# cols_must_not_zero = ["Nama","Usia","Gaji","Jenis Kelamin"]


# Tandai 0 sebagai NaN pada kolom yang tak mungkin 0
df_fixed = df.copy()
df_fixed[cols_must_not_zero] = df_fixed[cols_must_not_zero].replace(0, np.nan)

# Cek ulang missing (sekarang akan muncul batang di beberapa kolom)
print(df_fixed.isna().sum())


# --- RINGKASAN MISSING ---
missing_counts = df_fixed.isna().sum().sort_values(ascending=False)
missing_pct = (df_fixed.isna().mean().round(4) * 100).sort_values(ascending=False)

summary = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_pct": missing_pct
})
print("\nRingkasan missing per kolom:")
print(summary)

# --- GRAFIK: JUMLAH MISSING PER KOLOM ---
plt.figure(figsize=(10, 5))
missing_counts.plot(kind='bar')
plt.title('Jumlah Missing per Kolom')
plt.xlabel('Kolom')
plt.ylabel('Jumlah Missing')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- GRAFIK: PERSENTASE MISSING PER KOLOM ---
plt.figure(figsize=(10, 5))
missing_pct.plot(kind='bar')
plt.title('Persentase Missing per Kolom (%)')
plt.xlabel('Kolom')
plt.ylabel('Persentase (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- OPSIONAL: PETA MISSING (BARIS x KOLOM) ---
plt.figure(figsize=(10, 6))
plt.imshow(df_fixed.isna(), aspect='auto', interpolation='nearest')
plt.title('Peta Missing per Baris/Kolom')
plt.xlabel('Kolom')
plt.ylabel('Baris')
plt.yticks([])  # baris biasanya banyak, jadi disembunyikan
plt.xticks(ticks=np.arange(len(df_fixed.columns)),
           labels=df_fixed.columns, rotation=45, ha='right')
plt.tight_layout()
plt.show()


