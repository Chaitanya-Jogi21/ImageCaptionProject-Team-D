# train.py

import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# -----------------------------
# Step 1: Extract features from images
# -----------------------------
def extract_features(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(model.input, model.layers[-2].output)  # Remove final classification layer
    features = {}
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = load_img(img_path, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        feature = model.predict(img_array, verbose=0)
        features[img_name] = feature
    np.save('photos_features.npy', features)
    return features

# -----------------------------
# Step 2: Load descriptions
# -----------------------------
def load_descriptions(file_path):
    descriptions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            img_id, img_desc = tokens[0], tokens[1:]
            # Ensure the key matches actual image filenames
            img_id = img_id.split('.')[0] + '.jpg'
            desc = 'startseq ' + ' '.join(img_desc) + ' endseq'
            if img_id not in descriptions:
                descriptions[img_id] = []
            descriptions[img_id].append(desc)
    np.save('descriptions.npy', descriptions)
    return descriptions




# -----------------------------
# Step 3: Create tokenizer
# -----------------------------
def create_tokenizer(descriptions):
    all_desc = []
    for key in descriptions:
        all_desc.extend(descriptions[key])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    return tokenizer

# -----------------------------
# Step 4: Prepare sequences for training
# -----------------------------
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = [], [], []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# -----------------------------
# Step 5: Define the model
# -----------------------------
def define_model(vocab_size, max_length):
    # Image feature extractor
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Paths
    image_dir = '../data/images/sample/'
       # Folder containing all images
    desc_file = '../descriptions.txt'  # Text file with captions
    
    # 1. Extract features if not already saved
    if not os.path.exists('photos_features.npy'):
        print("Extracting features from images...")
        photos = extract_features(image_dir)
    else:
        photos = np.load('photos_features.npy', allow_pickle=True).item()
    
    # 2. Load descriptions
    if not os.path.exists('descriptions.npy'):
        print("Loading descriptions...")
        descriptions = load_descriptions(desc_file)
    else:
        descriptions = np.load('descriptions.npy', allow_pickle=True).item()
    
    # 3. Create tokenizer
    if not os.path.exists('tokenizer.npy'):
        print("Creating tokenizer...")
        tokenizer = create_tokenizer(descriptions)
    else:
        tokenizer = np.load('tokenizer.npy', allow_pickle=True).item()
    
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(tokenizer.texts_to_sequences([desc])[0]) for key in descriptions for desc in descriptions[key])
    
    # 4. Prepare sequences
    print("Preparing sequences...")
    X1, X2, y = create_sequences(tokenizer, max_length, descriptions, photos, vocab_size)
    
    # 5. Define model
    print("Defining model...")
    model = define_model(vocab_size, max_length)
    print(model.summary())
    
    # 6. Train model
    checkpoint = ModelCheckpoint('model.keras', monitor='loss', verbose=1, save_best_only=True, mode='min')

    model.fit([X1, X2], y, epochs=10, batch_size=64, callbacks=[checkpoint])

    print("Available photo keys:", list(photos.keys())[:5])
    print("Available description keys:", list(descriptions.keys())[:5])

