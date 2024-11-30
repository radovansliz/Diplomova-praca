import pandas as pd
df = pd.read_csv('../AndMal2020_19_11_2024/static-analysis/Ben0.csv')

print(df.head())

names = df.columns

print(f"Names: {names}")