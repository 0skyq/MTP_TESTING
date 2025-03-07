import os
import sys
import math
import time
import random
import csv
import cv2
import math
import numpy as np
import pandas as pd
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
layers = tf.keras.layers
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

from parameters import*

@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):

    def __init__(self,name = 'ENCODER',**kwargs):
        super().__init__(name = name ,**kwargs)

        self.latent_dim = LATENT_DIM

        self.conv1 = layers.Conv2D(32, (4, 4), activation='relu', strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(128, (4, 4), activation='relu', strides=2, padding='same')
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu', strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1024, activation='relu')
        self.mu = layers.Dense(self.latent_dim)
        self.sigma = layers.Dense(self.latent_dim)

        self.N = tfp.distributions.Normal(loc=0.0, scale=1.0)
        self.kl = 0  

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)

        mu = self.mu(x)

        sigma = tf.exp(self.sigma(x))  
        z = mu + sigma * tfd.Normal(0.0, 1.0).sample(tf.shape(mu))  
        self.kl = tf.reduce_sum(sigma**2 + mu**2 - tf.math.log(sigma) - 0.5) 

        return z


@tf.keras.utils.register_keras_serializable()
class Actor(tf.keras.Model):

    def __init__(self,name = 'ACTOR',**kwargs):
        super().__init__(name = name ,**kwargs)

        self.obs_dim = OBSERVATION_DIM
        self.action_dim = ACTION_DIM
        self.action_std_init = ACTION_STD_INIT

        self.dense1 = layers.Dense(500, activation='tanh')
        self.dense2 = layers.Dense(300, activation='tanh')
        self.dense3 = layers.Dense(100, activation='tanh')
        self.output_layer = layers.Dense(self.action_dim, activation='tanh')


    def call(self, obs):

        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        if len(tf.shape(obs)) == 1:
            obs = tf.expand_dims(obs, axis=0)

        obs = self.normalize(obs)

        x = self.dense1(obs)
        x = self.dense2(x)
        x = self.dense3(x)
        mean = self.output_layer(x)

        return mean

    def normalize(self, obs):

        obs = tf.clip_by_value(obs, clip_value_min=-1e8, clip_value_max=1e8)
        return obs



def data_processing(episode_id):

    images_folder_path = os.path.join(TEST_IMAGES, f"Episode_images_{episode_id}")
    csv_file_path = os.path.join(TEST_IMAGES, f"Episode_data_{episode_id}.csv")

    if not os.path.exists(images_folder_path):
        print(f"Images folder {images_folder_path} not found. Skipping.")
        sys.exit()
    if not os.path.exists(csv_file_path):
        print(f"CSV file {csv_file_path} not found. Skipping.")
        sys.exit()

    data = pd.read_csv(csv_file_path)

    observation = []
    cal_mean = []
    cal_reward = []
    gpu_time = []

    for index, row in data.iterrows():

        image_name = f"{row['sr.no']}.png"  
        image_path = os.path.join(images_folder_path, image_name)

        if not os.path.exists(image_path):
            print(f"Image {image_name} not found in {images_folder_path}. Skipping.")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image {image_name}. Skipping.")
            continue


        navgation_data = [row['throttle'],row['velocity'],row['norm_velocity'],row['nor_dis_center'],row['nor_angle']]

        cal_mean.append([row['mean[0]'],row['mean[1]']])     
        cal_reward.append(row['reward'])   
        gpu_time.append(row['exe_time'])

        observation.append([image,navgation_data])


    return observation,cal_mean,cal_reward,gpu_time
    



def accuracy_check_32bit():

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    encoder = Encoder()

    encoder = tf.keras.models.load_model(VAE_MODEL_PATH+'/var_auto_encoder_model')

    print(f'Variational AutoEncoder Loaded from {VAE_MODEL_PATH}')

    agent = Actor()
    agent = tf.keras.models.load_model(PPO_MODEL_PATH + '/actor')
    print(f"actor Model is  loaded from {PPO_MODEL_PATH}")
    print()


    for epiosde_id in range(1,NO_OF_TEST_EPISODES):

        print(f"EPISODE : {epiosde_id}")

        obs,cal_mean,cal_reward,gpu_time = data_processing(epiosde_id)

        print(len(obs))

        pred_mean = []
        pred_reward = []
        cpu_time = []


        for row in obs:

            s_time = time.time()

            image_obs = tf.convert_to_tensor(row[0], dtype=tf.float32)
            image_obs = tf.expand_dims(image_obs, axis=0)  

            obs_1 = encoder(image_obs)
            observation = tf.concat([tf.reshape(tf.cast(obs_1, tf.float32), [-1]),tf.cast(row[1], tf.float32)], axis=-1)


            mean = agent(observation).numpy().flatten()

            e_time = time.time()

            pred_mean.append([mean[0],mean[1]])

            cpu_time.append(e_time-s_time)


        loss = np.mean((np.array(pred_mean) - np.array(cal_mean)) ** 2)
        rsme = math.sqrt(loss)

        print(f"gpu time {np.array(np.mean(gpu_time))}")
        print(f"cpu_time {np.array(np.mean(cpu_time))}")
        print(f"MSE is {loss}")
        print(f"RSME is {rsme}")
        print()


        save_csv_dir = os.path.join(RESULTS_PATH,f'accuracy_check_32bit_pi.csv')

        gpu_time = np.mean(gpu_time)
        cpu_time = np.mean(cpu_time)
        MSE = loss
        RMSE = rsme

        data_to_append = [f"Episode_{epiosde_id}", gpu_time, cpu_time, MSE, RMSE]

        with open(save_csv_dir, mode='a', newline='') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow(["Episode", "GPU Time", "CPU Time", "MSE", "RMSE"])
            
            writer.writerow(data_to_append)

    
    print(f"Data saved to {save_csv_dir}")
        



def accuracy_check_16bit():

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

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


    for epiosde_id in range(1,NO_OF_TEST_EPISODES):

        print(f"EPISODE : {epiosde_id}")

        obs,cal_mean,cal_reward,gpu_time = data_processing(epiosde_id)

        #cal_mean = np.array(cal_mean, dtype=np.float16)


        print(len(obs))

        pred_mean = []
        pred_reward = []
        cpu_time = []

        for row in obs:

            s_time = time.time()

            image_obs = tf.convert_to_tensor(row[0], dtype=tf.float32)
            image_obs = tf.expand_dims(image_obs, axis=0)  


            encoder.set_tensor(encoder_input_details[0]['index'], image_obs)
            encoder.invoke()
            tflite_output = encoder.get_tensor(encoder_output_details[0]['index'])
            observation = tf.concat([tf.reshape(tf.cast(tflite_output, tf.float32), [-1]),tf.cast(row[1], tf.float32)], axis=-1)
            
            observation = tf.expand_dims(observation, axis=0) 

            agent.set_tensor(agent_input_details[0]['index'], observation)
            agent.invoke()
            mean = agent.get_tensor(agent_output_details[0]['index'])[0]

            e_time = time.time()

            pred_mean.append([mean[0],mean[1]])

            cpu_time.append(e_time-s_time)


        loss = np.mean((np.array(pred_mean) - np.array(cal_mean)) ** 2)
        rsme = math.sqrt(loss)

        print(f"gpu time {np.array(np.mean(gpu_time))}")
        print(f"cpu_time {np.array(np.mean(cpu_time))}")
        print(f"MSE is {loss}")
        print(f"RSME is {rsme}")
        print()


        save_csv_dir = os.path.join(RESULTS_PATH,f'accuracy_check_16bit_pi.csv')

        gpu_time = np.mean(gpu_time)
        cpu_time = np.mean(cpu_time)
        MSE = loss
        RMSE = rsme

        data_to_append = [f"Episode_{epiosde_id}", gpu_time, cpu_time, MSE, RMSE]

        with open(save_csv_dir, mode='a', newline='') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                writer.writerow(["Episode", "GPU Time", "CPU Time", "MSE", "RMSE"])
            
            writer.writerow(data_to_append)

    
    print(f"Data saved to {save_csv_dir}")
        


if __name__ == "__main__":
    try:
        accuracy_check_32bit()
        accuracy_check_16bit()


    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nTerminating...")