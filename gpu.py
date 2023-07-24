import tensorflow as tf
import subprocess

# Verify GPU availability and usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Available GPU(s):")
    for gpu in gpus:
        print(gpu)

    # Set TensorFlow to use GPU
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("TensorFlow is using GPU:", gpus[0])

    # Get GPU name using nvidia-smi
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE)
    gpu_name = result.stdout.decode().strip()
    print("GPU Name:", gpu_name)
else:
    print("No GPU found. TensorFlow is using CPU.")