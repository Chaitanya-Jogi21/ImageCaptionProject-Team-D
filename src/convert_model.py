from keras.models import load_model, save_model
from keras.utils import CustomObjectScope
from keras.layers import Layer

# Paths
old_model_path = 'model.h5'
new_model_path = 'model.keras'

# Custom layer (minimal fix)
class NotEqual(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # accept Keras kwargs like 'name'

    def call(self, inputs):
        x, y = inputs
        return tf.not_equal(x, y)

# Convert only if .keras doesn't exist
if not os.path.exists(new_model_path):
    with CustomObjectScope({'NotEqual': NotEqual}):
        model = load_model(old_model_path, compile=False)
    save_model(model, new_model_path)
    print("Conversion complete.")
else:
    print(".keras model already exists. Skipping conversion.")
