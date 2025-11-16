import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# Paths
IMAGE_DIR = "data/images/train2017"
FEATURE_DIR = "data/features"

os.makedirs(FEATURE_DIR, exist_ok=True)

# Load InceptionV3 model + remove top layer
model = InceptionV3(weights='imagenet')
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Function to extract features
def extract_features(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# Loop through images
image_files = os.listdir(IMAGE_DIR)
for img_file in tqdm(image_files):
    img_path = os.path.join(IMAGE_DIR, img_file)
    feature = extract_features(img_path)
    feature_file = os.path.join(FEATURE_DIR, img_file.split(".")[0] + ".npy")
    np.save(feature_file, feature)

print("✅ Feature extraction completed!")

import pickle

# Combine all extracted features into one dictionary
print("🔹 Combining all features into one dictionary...")
features_dict = {}
for img_file in tqdm(image_files, desc="Combining features"):
    img_id = img_file.split(".")[0]
    feature_path = os.path.join(FEATURE_DIR, img_id + ".npy")
    features_dict[img_id] = np.load(feature_path)

# Save combined features as a pickle file
os.makedirs("data/processed", exist_ok=True)
with open("data/processed/features.pkl", "wb") as f:
    pickle.dump(features_dict, f)

print("💾 Saved combined features to data/processed/features.pkl successfully!")

