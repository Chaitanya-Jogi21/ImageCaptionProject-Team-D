import pickle

with open("data/processed/features.pkl", "rb") as f:
    features = pickle.load(f)

with open("data/processed/captions.pkl", "rb") as f:
    captions = pickle.load(f)

print("Sample features keys:", list(features.keys())[:5])
print("Sample captions keys:", list(captions.keys())[:5])
