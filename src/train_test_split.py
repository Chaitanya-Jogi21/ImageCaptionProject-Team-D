import pickle
from sklearn.model_selection import train_test_split

# Load features and captions
with open("data/processed/features.pkl", "rb") as f:
    features = pickle.load(f)

with open("data/processed/captions.pkl", "rb") as f:
    captions = pickle.load(f)

# Convert captions keys to strings with leading zeros to match features keys
captions_str_keys = {str(k).zfill(12): v for k, v in captions.items()}

# Keep only image IDs present in both features and captions
common_ids = [img_id for img_id in captions_str_keys if img_id in features]

# Split into train/test
train_ids, test_ids = train_test_split(common_ids, test_size=0.2, random_state=42)

# Create train/test dictionaries
train_captions = {img: captions_str_keys[img] for img in train_ids}
test_captions = {img: captions_str_keys[img] for img in test_ids}

train_features = {img: features[img] for img in train_ids if img in features}
test_features = {img: features[img] for img in test_ids if img in features}

# Save
with open("data/processed/train_features.pkl", "wb") as f:
    pickle.dump(train_features, f)
with open("data/processed/test_features.pkl", "wb") as f:
    pickle.dump(test_features, f)

with open("data/processed/train_captions.pkl", "wb") as f:
    pickle.dump(train_captions, f)
with open("data/processed/test_captions.pkl", "wb") as f:
    pickle.dump(test_captions, f)

print("✅ Train/test split fixed!")
print("Train features:", len(train_features))
print("Test features:", len(test_features))
