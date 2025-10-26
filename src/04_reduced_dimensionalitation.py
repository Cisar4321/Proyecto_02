import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("data/features_combined.npy")
OUTPUT_PCA = Path("data/X_pca.npy")
OUTPUT_SVD = Path("data/X_svd.npy")

X = np.load(INPUT_PATH)
print("Shape original:", X.shape)

# --- PCA ---
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)
np.save(OUTPUT_PCA, X_pca)
print("Shape PCA:", X_pca.shape)
print("Varianza explicada acumulada (PCA):", np.sum(pca.explained_variance_ratio_))

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.title("PCA - Varianza explicada")
plt.grid(True)
plt.savefig("data/pca_varianza.png", dpi=300)
plt.close()

# --- SVD ---
svd = TruncatedSVD(n_components=100)
X_svd = svd.fit_transform(X)
np.save(OUTPUT_SVD, X_svd)
print("Shape SVD:", X_svd.shape)
print("Varianza explicada acumulada (SVD):", np.sum(svd.explained_variance_ratio_))

plt.figure()
plt.plot(np.cumsum(svd.explained_variance_ratio_))
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.title("SVD - Varianza explicada")
plt.grid(True)
plt.savefig("data/svd_varianza.png", dpi=300)
plt.close()

print("\nArchivos generados:")
print("- data/X_pca.npy")
print("- data/X_svd.npy")
print("- data/pca_varianza.png")
print("- data/svd_varianza.png")
