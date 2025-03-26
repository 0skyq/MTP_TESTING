import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
layers = tf.keras.layers

from parameters import*


def representative_data_gen_vae():
    for _ in range(100):
        dummy_input = tf.random.uniform([1, 160, 80, 3], minval=0.0, maxval=128.0)
        yield [dummy_input]

def representative_data_gen_actor():
    for _ in range(100):
        dummy_input = tf.random.uniform([1, 100], minval=-1.0, maxval=1.0) 
        yield [dummy_input]


def convert_tflite():

    
    if not os.path.exists(TF_LITE_PATH):
            os.makedirs(TF_LITE_PATH)

    models = {
        "actor": os.path.join(PPO_MODEL_PATH, "actor"),
        "var_auto_encoder_model": os.path.join(VAE_MODEL_PATH, "var_auto_encoder_model")
    }

    for model_name, model_path in models.items():
        model = tf.keras.models.load_model(model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()


        output_path = os.path.join(TF_LITE_PATH, f"{model_name}_fp16.tflite")
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        print(f"Converted model saved at {output_path}")

        # INT8 conversion

        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        if model_name == "var_auto_encoder_model":
            converter_int8.representative_dataset = representative_data_gen_vae
        elif model_name == "actor":
            converter_int8.representative_dataset = representative_data_gen_actor

        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.uint8
        converter_int8.inference_output_type = tf.uint8
        tflite_model_int8 = converter_int8.convert()
        
        output_int8 = os.path.join(TF_LITE_PATH, f"{model_name}_int8.tflite")
        with open(output_int8, "wb") as f:
            f.write(tflite_model_int8)
        print(f"INT8 converted model saved at {output_int8}")





if __name__ == "__main__":
    try:
        convert_tflite()

    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nTerminating...")