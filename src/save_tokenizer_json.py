import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the tokenizer saved as .npy
tokenizer = np.load('tokenizer.npy', allow_pickle=True).item()

# Convert tokenizer to JSON
tokenizer_json = tokenizer.to_json()

# Save as tokenizer.json
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)

print("tokenizer.json saved successfully!")
