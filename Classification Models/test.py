import pandas as pd

# Načítanie súboru
file_path = 'cicandmal2020-dynamic.parquet'
df = pd.read_parquet(file_path)

# Zobrazenie základných informácií o dátach
print("Základné informácie o dátach:")
print(df.info())

# Zobrazenie základných štatistík
print("\nŠtatistiky dát:")
print(df.describe())

# Ukážka niekoľkých riadkov
print("\nUkážka niekoľkých riadkov:")
print(df.head())

# Skontrolovanie chýbajúcich hodnôt
print("\nChýbajúce hodnoty v dátach:")
print(df.isnull().sum())

# Počet riadkov a stĺpcov
print(f"\nPočet riadkov: {df.shape[0]}")
print(f"Počet stĺpcov: {df.shape[1]}")