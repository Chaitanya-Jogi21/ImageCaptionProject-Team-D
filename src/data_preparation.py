import os
import numpy as np
import pandas as pd
import string
import json
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from PIL import Image
import nltk

# Ensure nltk punkt is available
nltk.download('punkt')

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images", "train2017")
CAPTIONS_DIR = os.path.join(DATA_DIR, "captions")
ANNOTATION_FILE = os.path.join(CAPTIONS_DIR, "captions_train2017.json")

# Output directories
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Step 1: Load Captions JSON
def load_captions():
    print("🔹 Loading captions JSON file...")
    with open(ANNOTATION_FILE, 'r') as f:
        data = json.load(f)
    captions_dict = {}
    for item in data['annotations']:
        image_id = item['image_id']
        caption = item['caption']
        captions_dict.setdefault(image_id, []).append(caption)
    print(f"✅ Loaded {len(captions_dict)} image captions.")
    return captions_dict

# Step 2: Preprocess Images
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = image / 255.0
    return image

# Step 3: Clean Captions
def clean_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# Step 4: Prepare Captions and Tokenizer
def prepare_captions(captions_dict):
    print("🔹 Cleaning and tokenizing captions...")
    all_captions = []
    for caption_list in captions_dict.values():
        for c in caption_list:
            c = clean_text(c)
            c = f"<start> {c} <end>"
            all_captions.append(c)

    tokenizer = Tokenizer(num_words=10000, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    sequences = tokenizer.texts_to_sequences(all_captions)
    padded = pad_sequences(sequences, padding='post')

    np.save(os.path.join(PROCESSED_DIR, "captions_padded.npy"), padded)
    with open(os.path.join(PROCESSED_DIR, "tokenizer.json"), "w") as f:
        f.write(json.dumps(tokenizer.to_json()))

    print(f"✅ Captions processed: {len(all_captions)}")
    return tokenizer, padded

# Step 5: Process Images and Save as Numpy

def process_images(batch_size=1000):
    """
    Process images in batches and save each batch separately.
    """
    print("🔹 Processing and normalizing images in batches...")
    
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
    total_images = len(image_files)
    print(f"Total images: {total_images}")

    for i in range(0, total_images, batch_size):
        batch_files = image_files[i:i+batch_size]
        image_arrays = []

        for file in tqdm(batch_files, desc=f"Processing batch {i//batch_size+1}"):
            img_path = os.path.join(IMAGES_DIR, file)
            try:
                img_array = preprocess_image(img_path)
                image_arrays.append(img_array)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        # Convert batch to numpy array
        image_arrays = np.array(image_arrays)

        # Save batch
        batch_file = os.path.join(PROCESSED_DIR, f"images_batch_{i//batch_size+1}.npy")
        np.save(batch_file, image_arrays)
        print(f"✅ Saved batch {i//batch_size+1} with {len(image_arrays)} images")

# Run all steps
if __name__ == "__main__":
    print("🚀 Starting Data Preparation...")
    captions_dict = load_captions()
    tokenizer, padded = prepare_captions(captions_dict)
    process_images()
    print("🎉 Data preparation completed successfully!")
