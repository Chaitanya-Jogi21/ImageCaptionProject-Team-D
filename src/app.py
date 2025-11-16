import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import CustomObjectScope
import tensorflow as tf
from keras.layers import Layer
import random

# -----------------------------
# Custom layer definition
# -----------------------------
class NotEqual(Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.not_equal(x, y)

# -----------------------------
# Load model
# -----------------------------
model_path = 'model.keras'
with CustomObjectScope({'NotEqual': NotEqual}):
    model = load_model(model_path, compile=False)
print("Model loaded successfully!")

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = np.load('tokenizer.npy', allow_pickle=True).item()
print("Tokenizer loaded successfully!")

# -----------------------------
# Load photo features
# -----------------------------
photos_features = np.load('photos_features.npy', allow_pickle=True)
print(f"Loaded {len(photos_features)} photos features.")

# -----------------------------
# Caption generation function
# -----------------------------
def generate_caption(model, tokenizer, photo, max_length=34):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq ', '').replace(' endseq', '')

# -----------------------------
# Pick a random image and generate caption
# -----------------------------
random_index = random.randint(0, len(photos_features)-1)
photo = photos_features[random_index].reshape(1, -1)  # reshape for model input
caption = generate_caption(model, tokenizer, photo)

print(f"Random image index: {random_index}")
print("Generated Caption:", caption)
