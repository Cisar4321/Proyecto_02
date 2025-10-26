import pandas as pd

df = pd.read_csv("data/movies_train.csv")

print("Dimensiones:", df.shape)
print("\nColumnas del CSV:")
print(df.columns.tolist())

print("\nPrimeras filas:")
print(df.head())

print("\nValores nulos por columna:")
print(df.isna().sum())
