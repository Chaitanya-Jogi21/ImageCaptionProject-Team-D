# src/predict_image_caption.py
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# ======================
# Load trained model
# ======================
model = load_model("models/best_model.h5")

# ======================
# Load tokenizer
# ======================
with open("data/processed/train_captions.pkl", "rb") as f:
    train_captions = pickle.load(f)

all_captions = [cap for caps in train_captions.values() for cap in caps]
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(seq.split()) for seq in all_captions)

# ======================
# Load image feature extractor
# ======================
inception_model = InceptionV3(weights='imagenet')
from tensorflow.keras.models import Model
feature_model = Model(inception_model.input, inception_model.layers[-2].output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = feature_model.predict(x, verbose=0)
    return feature

# ======================
# Generate caption
# ======================
def generate_caption(model, tokenizer, feature, max_length):
    in_text = "<start>"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
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
img_path = "data/images/example.jpg"  # put your image path here
feature = extract_features(img_path)
caption = generate_caption(model, tokenizer, feature, max_length)
print("Generated caption:", caption)
