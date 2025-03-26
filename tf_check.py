import tensorflow as tf
print("TensorFlow Version:", tf.__version__)

gpu_available = tf.config.list_physical_devices('GPU')
if gpu_available:
    print("CUDA is available! TensorFlow is using the GPU.")
    for gpu in gpu_available:
        print("GPU:", gpu)
else:
    print("CUDA is NOT available. TensorFlow is using the CPU.")

