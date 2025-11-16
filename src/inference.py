# inference.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# ---------------------------
# Load the trained model
# ---------------------------
model = load_model('model.keras')

# ---------------------------
# Load the tokenizer
# ---------------------------
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

max_length = 8  # Use the same max_length used in training

# ---------------------------
# Feature extraction
# ---------------------------
def extract_feature(image_path):
    model_incep = InceptionV3(weights='imagenet')
    model_incep = tf.keras.Model(model_incep.input, model_incep.layers[-2].output)
    
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    feature = model_incep.predict(img_array, verbose=0)
    return feature

# ---------------------------
# Generate caption
# ---------------------------
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    # Clean the caption
    caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    # Remove repeated consecutive words
    caption_words = caption.split()
    final_caption = []
    for i, word in enumerate(caption_words):
        if i == 0 or word != caption_words[i-1]:
            final_caption.append(word)
    return ' '.join(final_caption)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    image_path = input("Enter the path of the image you want to caption: ").strip()
    try:
        photo_feature = extract_feature(image_path)
        caption = generate_caption(model, tokenizer, photo_feature, max_length)
        print("Generated Caption:", caption)
    except FileNotFoundError:
        print("Error: Image not found. Please check the path.")
