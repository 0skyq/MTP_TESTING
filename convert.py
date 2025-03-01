import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
layers = tf.keras.layers

from parameters import*

DTYPE = tf.float16


def convert_tflite():

    
    if not os.path.exists(TF_LITE_PATH):
            os.makedirs(TF_LITE_PATH)

    models = {
        "actor": os.path.join(PPO_MODEL_PATH, "actor"),
        "critic": os.path.join(PPO_MODEL_PATH, "critic"),
        "var_auto_encoder_model": os.path.join(VAE_MODEL_PATH, "var_auto_encoder_model")
    }

    for model_name, model_path in models.items():
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [DTYPE]
        tflite_model = converter.convert()


        output_path = os.path.join(TF_LITE_PATH, f"{model_name}_fp16.tflite")
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        print(f"Converted model saved at {output_path}")


if __name__ == "__main__":
    try:
        convert_tflite()

    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nTerminating...")