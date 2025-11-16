import os
import numpy as np
import pickle
from tqdm import tqdm

# Paths
FEATURE_DIR = "data/features"
OUTPUT_FILE = "data/processed/features.pkl"

# Dictionary to hold all features
features = {}

# Loop through all .npy files in features folder
files = os.listdir(FEATURE_DIR)
for f in tqdm(files, desc="Combining features"):
    # Remove extension from filename to match image IDs in captions
    # Example: '228085.jpg.npy' -> '228085'
    img_id = f.split('.')[0]  
    features[img_id] = np.load(os.path.join(FEATURE_DIR, f))

# Make sure processed folder exists
os.makedirs("data/processed", exist_ok=True)

# Save combined features
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(features, f)

print(f"✅ All features combined and saved to {OUTPUT_FILE}")
