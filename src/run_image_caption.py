import numpy as np
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.layers import Layer

# Paths
new_model_path = 'model.keras'
tokenizer_path = 'tokenizer.npy'
photos_features_path = 'photos_features.npy'

# Minimal NotEqual fix
class NotEqual(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        x, y = inputs
        return tf.not_equal(x, y)

# Load model safely
with CustomObjectScope({'NotEqual': NotEqual}):
    model = load_model(new_model_path, compile=False)
print("Model loaded successfully!")

# Load tokenizer and photo features (your existing code)
tokenizer = np.load(tokenizer_path, allow_pickle=True).item()
photos_features = np.load(photos_features_path, allow_pickle=True)
