import tensorflow as tf

# List all available physical devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs found: {gpus}")
else:
    print("No GPUs found.")
