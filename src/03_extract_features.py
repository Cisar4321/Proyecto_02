import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.feature import local_binary_pattern, hog

DATA_PATH = Path("data/movies_with_posters.csv")
POSTERS_DIR = Path("data/posters")
OUTPUT_NPY = Path("data/features_combined.npy")
OUTPUT_CSV = Path("data/features_combined.csv")

def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_lbp_features(gray, num_points=24, radius=3):
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, num_points + 3),
                           range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_hog_features(gray):
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=False,
        feature_vector=True
    )
    return features

def extract_hu_moments(gray):
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

df = pd.read_csv(DATA_PATH)
features = []
titles = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = Path(row["poster_path"])
    image = cv2.imread(str(path))
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist_color = extract_color_histogram(image)
    hist_lbp = extract_lbp_features(gray)
    hist_hog = extract_hog_features(gray)
    hu = extract_hu_moments(gray)

    feature_vector = np.hstack([hist_color, hist_lbp, hist_hog, hu])

    features.append(feature_vector)
    titles.append(row["Title"])

X = np.array(features)
np.save(OUTPUT_NPY, X)
pd.DataFrame(X, index=titles).to_csv(OUTPUT_CSV)

print("\nGuardado:")
print(f"- {OUTPUT_NPY}")
print(f"- {OUTPUT_CSV}")
print("Shape de matriz de características:", X.shape)
