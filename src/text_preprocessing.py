import os
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Paths
CAPTIONS_FILE = "data/annotations/captions_train2017.json"
OUTPUT_DIR = "data/tokenizer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load captions
print("🔹 Loading captions...")
with open(CAPTIONS_FILE, 'r') as f:
    captions_data = json.load(f)

captions_dict = {}
for item in captions_data['annotations']:
    image_id = item['image_id']
    caption = item['caption'].lower().replace('.', '').replace(',', '')
    caption = 'startseq ' + caption + ' endseq'
    captions_dict.setdefault(image_id, []).append(caption)

# Combine all captions
all_captions = [caption for captions in captions_dict.values() for caption in captions]

# Tokenize text
print("🔹 Tokenizing captions...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

# Save tokenizer
with open(os.path.join(OUTPUT_DIR, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)

# Convert to sequences
max_length = max(len(caption.split()) for caption in all_captions)
print(f"✅ Vocabulary Size: {vocab_size}")
print(f"✅ Max Caption Length: {max_length}")

print("🎉 Caption tokenization completed successfully!")

# Make sure the processed data folder exists
os.makedirs("data/processed", exist_ok=True)

# Save tokenized captions and tokenizer
with open("data/processed/captions.pkl", "wb") as f:
    pickle.dump(captions_dict, f)  # ✅ your variable name is 'captions_dict'

with open("data/processed/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("💾 Saved captions_dict and tokenizer to 'data/processed/' successfully!")