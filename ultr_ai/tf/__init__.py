import os

# The transformers library does not support Keras v3, so we have to force the use of legacy Keras.
os.environ['TF_USE_LEGACY_KERAS'] = '1'