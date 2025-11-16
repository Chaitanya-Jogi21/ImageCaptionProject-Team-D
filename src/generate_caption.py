# src/generate_caption.py
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ======================
# Load model
# ======================
model = load_model("models/best_model.h5")

# ======================
# Load tokenizer
# ======================
with open("data/processed/train_captions.pkl", "rb") as f:
    train_captions = pickle.load(f)

# Recreate tokenizer (must match training)
all_captions = [cap for caps in train_captions.values() for cap in caps]
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(seq.split()) for seq in all_captions)

# ======================
# Load image features
# ======================
with open("data/processed/features.pkl", "rb") as f:
    features = pickle.load(f)

# ======================
# Generate caption function
# ======================
def generate_caption(model, tokenizer, image_id, features, max_length):
    in_text = "<start>"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([features[image_id], sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += " " + word
        if word == "<end>":
            break
    return in_text.replace("<start>", "").replace("<end>", "").strip()

# ======================
# Example usage
# ======================
image_id = list(features.keys())[0]  # pick first image
caption = generate_caption(model, tokenizer, image_id, features, max_length)
print("Generated caption:", caption)
