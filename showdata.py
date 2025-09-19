import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 0) Load data
URL = "diabetes.csv"
df = pd.read_csv(URL)


# 1A) Ukuran dataset (baris, kolom)
print("Shape:", df.shape)

# 1B) Lima data teratas
print("\nLima Data Teratas:")
print(df.head())

# 1C) Info kolom: tipe data & jumlah non-null
print("\nInfo Dataset:")
print(df.info())

# 1D) Statistik deskriptif numerik
print("\nStatistik Deskriptif (numerik):")
print(df.describe())
