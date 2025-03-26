import os
import sys
import random
import socket
import struct
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
layers = tf.keras.layers
from parameters import*
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((SIMULATION_IP,PORT))
print("Connection Established")



def data_processing_16bit():

    header = client_socket.recv(12)
    h,w,c = struct.unpack("3I",header)
    info_size = 5
    image_size = h*w*c

    image_bytes = b""

    while len(image_bytes)<image_size:
        image_bytes += client_socket.recv(image_size - len(image_bytes))

    info_bytes = client_socket.recv(info_size*4)

    image_array = np.frombuffer(image_bytes,dtype = np.uint8).reshape((h,w,c))
    info_array = np.frombuffer(info_bytes,dtype=np.float32)


    return image_array,info_array



def run_16bit():


    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    action_dim = ACTION_DIM
    action_std_init = ACTION_STD_INIT

    encoder = tf.lite.Interpreter(model_path=TF_LITE_PATH + "/var_auto_encoder_model_fp16.tflite")
    encoder.allocate_tensors()

    encoder_input_details = encoder.get_input_details()
    encoder_output_details = encoder.get_output_details()

    agent = tf.lite.Interpreter(model_path=TF_LITE_PATH + "/actor_fp16.tflite")



    agent.allocate_tensors()

    agent_input_details = agent.get_input_details()
    agent_output_details = agent.get_output_details()

    print(f'Lite models loaded from {TF_LITE_PATH}')
    print(encoder_input_details[0]['dtype'])  
    print(agent_input_details[0]['dtype'])  


    while True:

        image_obs , info_obs = data_processing_16bit()

        image_obs = tf.convert_to_tensor(image_obs, dtype=tf.float32)
        image_obs = tf.expand_dims(image_obs, axis=0)  


        encoder.set_tensor(encoder_input_details[0]['index'], image_obs)
        encoder.invoke()
        tflite_output = encoder.get_tensor(encoder_output_details[0]['index'])
        observation = tf.concat([tf.reshape(tf.cast(tflite_output, tf.float32), [-1]),tf.cast(info_obs, tf.float32)], axis=-1)
        
        observation = tf.expand_dims(observation, axis=0) 

        agent.set_tensor(agent_input_details[0]['index'], observation)
        agent.invoke()
        mean = agent.get_tensor(agent_output_details[0]['index'])[0]

        log_std = tf.Variable(tf.fill((action_dim,), action_std_init), trainable=False, dtype=tf.float32)
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=log_std)
        action = dist.sample()

        action = np.array(action,dtype = np.float32).flatten()

        print(action)

        data = struct.pack('2f',*action)

        client_socket.sendall(data)
        #print("action sent")


    client_socket.close()

    sys.exit()



        


if __name__ == "__main__":
    run()

