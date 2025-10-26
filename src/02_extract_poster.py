"""
02_download_posters.py — Descarga los pósters del dataset MovieGenre.csv
Guarda las imágenes en data/posters/ y crea un CSV limpio con la ruta local.
"""

import os
import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path

DATA_PATH = Path("data/MovieGenre.csv")
OUTPUT_DIR = Path("data/posters")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = Path("data/movies_with_posters.csv")

df = pd.read_csv(DATA_PATH, encoding="latin1")

if not {"Title", "Genre", "Poster"}.issubset(df.columns):
    raise ValueError("El CSV debe contener las columnas 'Title', 'Genre' y 'Poster'")

df = df.dropna(subset=["Poster"])
print(f"Películas con póster válido: {len(df)}")

def download_image(row):
    title = str(row["Title"])
    url = str(row["Poster"]).strip()
    filename = (
        title.replace(" ", "_")
             .replace("/", "_")
             .replace(":", "_")
             .replace("(", "")
             .replace(")", "")
             .replace("|", "_")
             .replace("'", "")
             .replace('"', "")
             .replace("?", "")
             .replace("!", "")
    ) + ".jpg"
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        return str(filepath)
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.headers["Content-Type"].startswith("image"):
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return str(filepath)
    except Exception as e:
        print(f"Error con {title}: {e}")
    return None

df_sample = df.head(500)
paths = []
for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    path = download_image(row)
    paths.append(path)

df_sample["poster_path"] = paths
df_clean = df_sample.dropna(subset=["poster_path"])
df_clean[["Title", "Genre", "poster_path"]].to_csv(OUTPUT_CSV, index=False)
print(f"\nGuardado: {OUTPUT_CSV}")
print(f"Total de imágenes descargadas: {len(df_clean)}")
