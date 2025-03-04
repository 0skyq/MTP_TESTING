import os
import sys
import glob
import math
import weakref
import pygame
import time
import random
import csv
import cv2
import pickle
import math
import socket
import struct
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
layers = tf.keras.layers
from parameters import*
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


class EncodeState:
    def __init__(self):
        self.model_path = os.path.join(VAR_AUTO_MODEL_PATH,'var_auto_encoder_model') 
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.model.trainable = False  


        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.training = False  
        
        print()
        print(f'Variational AutoEncoder Loaded from {self.model_path}')
        print()

    def process(self, observation):
        image_obs = tf.convert_to_tensor(observation[0], dtype=tf.float32)
        image_obs = tf.expand_dims(image_obs, axis=0)  
        image_obs = self.model(image_obs, training=False)  
        navigation_obs = tf.convert_to_tensor(observation[1], dtype=tf.float32)
        observation = tf.concat([tf.reshape(image_obs, [-1]), navigation_obs], axis=-1)

        return observation


class Buffer:
    def __init__(self):
        self.observation = []  
        self.actions = []         
        self.log_probs = []     
        self.rewards = []         
        self.dones = []

    def clear(self):
        del self.observation[:]    
        del self.actions[:]        
        del self.log_probs[:]      
        del self.rewards[:]
        del self.dones[:]




@tf.keras.utils.register_keras_serializable()
class Actor(tf.keras.Model):

    def __init__(self,name = 'ACTOR',**kwargs):
        super().__init__(name = name ,**kwargs)

        self.obs_dim = OBSERVATION_DIM
        self.action_dim = ACTION_DIM
        self.action_std_init = ACTION_STD_INIT

        # self.dense1 = layers.Dense(500, activation='tanh',kernel_initializer= 'glorot_uniform')
        # self.dense2 = layers.Dense(300, activation='tanh',kernel_initializer= 'glorot_uniform')
        # self.dense3 = layers.Dense(100, activation='tanh',kernel_initializer= 'glorot_uniform')
        # self.output_layer = layers.Dense(self.action_dim, activation='tanh',kernel_initializer= 'glorot_uniform')

        
        self.dense1 = layers.Dense(500, activation='tanh')
        self.dense2 = layers.Dense(300, activation='tanh')
        self.dense3 = layers.Dense(100, activation='tanh')
        self.output_layer = layers.Dense(self.action_dim, activation='tanh')

    def call(self, obs):

        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        
        obs = self.normalize(obs)

        x = self.dense1(obs)
        x = self.dense2(x)
        x = self.dense3(x)
        mean = self.output_layer(x)

        return mean

    def normalize(self, obs):

        obs = tf.clip_by_value(obs, clip_value_min=-1e8, clip_value_max=1e8)
        return obs




@tf.keras.utils.register_keras_serializable()
class Critic(tf.keras.Model):

    def __init__(self,name = 'CRITIC',**kwargs):
        super().__init__(name = name ,**kwargs)

        # self.dense1 = layers.Dense(500, activation='tanh',kernel_initializer= 'glorot_uniform')
        # self.dense2 = layers.Dense(300, activation='tanh',kernel_initializer= 'glorot_uniform')
        # self.dense3 = layers.Dense(100, activation='tanh',kernel_initializer= 'glorot_uniform')
        # self.output_layer = layers.Dense(1,kernel_initializer= 'glorot_uniform')

        self.dense1 = layers.Dense(500, activation='tanh')
        self.dense2 = layers.Dense(300, activation='tanh')
        self.dense3 = layers.Dense(100, activation='tanh')
        self.output_layer = layers.Dense(1)


    def call(self, obs):

        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        obs = self.normalize(obs)

        x = self.dense1(obs)
        x = self.dense2(x)
        x = self.dense3(x)
        value = self.output_layer(x)

        return value
  
    def normalize(self, obs):

        obs = tf.clip_by_value(obs, clip_value_min=-1e8, clip_value_max=1e8)
        return obs





@tf.keras.utils.register_keras_serializable()
class PPOAgent(tf.keras.Model):
    

    def __init__(self, name="PPOAgent", **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.obs_dim = OBSERVATION_DIM
        self.action_dim = ACTION_DIM
        self.action_std_init = ACTION_STD_INIT
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.lam = LAMBDA  
        self.lr = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.n_updates_per_iteration = NO_OF_ITERATIONS
        self.memory = Buffer()
        self.town = TOWN
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss = tf.keras.losses.MeanSquaredError()

        self.models_dir = PPO_MODEL_PATH
        self.checkpoint_dir = CHECKPOINT_PATH

        self.log_std = tf.Variable(tf.fill((self.action_dim,), self.action_std_init), trainable=False, dtype=tf.float32)
        #self.log_std = tf.Variable(tf.fill((self.action_dim,), tf.math.log(self.action_std_init)), trainable=False, dtype=tf.float32)


        self.actor = Actor()
        self.critic = Critic()
        self.old_actor = Actor()
        self.old_critic = Critic()

        self.actor.compile(optimizer=self.optimizer)
        self.critic.compile(optimizer=self.optimizer)
        self.old_actor.compile(optimizer=self.optimizer)
        self.old_critic.compile(optimizer=self.optimizer)
        

        self.update_old_policy()

    
    def call(self, obs, train):

        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        if len(obs.shape) == 1:
            obs = tf.expand_dims(obs, axis=0) 
        
        mean = self.old_actor(obs)

        if tf.reduce_any(tf.math.is_nan(mean)):
            print("NaN detected in the mean, exiting...")
            exit()

        action, log_probs = self.get_action_and_log_prob(mean)
        value = self.old_critic(obs)

        if tf.reduce_any(tf.math.is_nan(value)):
            print("NaN detected in the value, exiting...")
            exit()
        
        if train:
            self.memory.observation.append(obs)
            self.memory.actions.append(action)
            self.memory.log_probs.append(log_probs)
        
        return action.numpy().flatten(),mean.numpy().flatten()


    def update_old_policy(self):
        self.old_actor.set_weights(self.actor.get_weights())
        self.old_critic.set_weights(self.critic.get_weights())
    

    def get_action_and_log_prob(self, mean):

        std = tf.exp(self.log_std)  
        #dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=self.log_std)

        #dist  = tfp.distributions.Normal(mean, tf.exp(self.log_std), validate_args=True)
        action = dist.sample()
        log_probs = dist.log_prob(action)

        return action, log_probs


    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        returns = advantages + values[:-1]
        #returns = tf.convert_to_tensor(advantages, dtype=tf.float32) + values[:-1]

        return tf.convert_to_tensor(advantages, dtype=tf.float32), tf.convert_to_tensor(returns, dtype=tf.float32)


    def evaluate(self, obs, action):

        mean = self.actor(obs)

        if tf.reduce_any(tf.math.is_nan(mean)):
            print("NaN detected in the mean, exiting...")
            exit()

        std = tf.exp(self.log_std)  
        #dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=self.log_std)

        #dist  = tfp.distributions.Normal(mean, tf.exp(self.log_std), validate_args=True)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        values = self.critic(obs)

        return log_probs, values, entropy


    def learn(self):
        print()
        rewards = self.memory.rewards
        dones = self.memory.dones
        old_states = tf.squeeze(tf.stack(self.memory.observation, axis=0))
        old_actions = tf.squeeze(tf.stack(self.memory.actions, axis=0))
        old_logprobs = tf.squeeze(tf.stack(self.memory.log_probs, axis=0))

        values = self.critic(old_states)
        values = tf.squeeze(values)
        values = tf.concat([values, tf.zeros((1,))], axis=0)

        advantages, returns = self.compute_advantages(rewards, values, dones)
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-7)
        returns = (returns - tf.reduce_mean(returns))/(tf.math.reduce_std(returns)+1e-7)
        # tf.keras.layers.LayerNormalization()(advantages)
        # tf.keras.layers.LayerNormalization()(returns)

        for i in range(self.n_updates_per_iteration):
            with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:

                log_probs, values, dist_entropy = self.evaluate(old_states, old_actions)
                values = tf.squeeze(values)
                #ratios = tf.exp(tf.clip_by_value(log_probs - old_logprobs, -10, 10))
                ratios = tf.exp(log_probs - old_logprobs)

                surr1 = ratios * advantages
                surr2 = tf.clip_by_value(ratios, 1 - self.clip, 1 + self.clip) * advantages

                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - 0.01 * tf.reduce_mean(dist_entropy)
                critic_loss = 0.5 * self.loss(values, returns)


            actor_vars = self.actor.trainable_variables #+ [self.log_std]  
            grads_a = tape_a.gradient(actor_loss, actor_vars)
            grads_c = tape_c.gradient(critic_loss, self.critic.trainable_variables)


            self.optimizer.apply_gradients(zip(grads_a, actor_vars))
            self.optimizer.apply_gradients(zip(grads_c, self.critic.trainable_variables))

            #print(f" A_Loss = {actor_loss.numpy():.6f}, C_Loss = {critic_loss.numpy():.6f},Entropy: {tf.reduce_mean(dist_entropy).numpy():.6f},Adv: {tf.reduce_mean(advantages).numpy()}")


        self.update_old_policy()
        self.memory.clear()

        print("\nUPDATED THE WEIGHTS\n")



    def save(self):

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.actor.save(self.models_dir + '/actor')
        self.critic.save(self.models_dir + '/critic')

        log_std_path = os.path.join(self.models_dir, 'log_std.npy')
        np.save(log_std_path, self.log_std.numpy())


        print(f"Model weights are saved at {self.models_dir}")


    def chkpt_save(self,episode,timestep,cumulative_score):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        checkpoint_file = os.path.join(self.checkpoint_dir ,'checkpoint.pickle')

        data = {
            'episode': episode,
            'timestep': timestep,
            'cumulative_score': cumulative_score,
            'log_std': self.log_std.numpy()
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"Checkpoint saved as {checkpoint_file}")
        print()


    def load(self):

        self.actor = tf.keras.models.load_model(self.models_dir + '/actor')
        self.critic = tf.keras.models.load_model(self.models_dir + '/critic')
        self.old_actor = tf.keras.models.load_model(self.models_dir + '/actor')
        self.old_critic = tf.keras.models.load_model(self.models_dir + '/critic')

        log_std_path = os.path.join(self.models_dir, 'log_std.npy')
        if os.path.exists(log_std_path):
            self.log_std.assign(np.load(log_std_path))

        print(f"Model is  loaded from {self.models_dir}")
        print()


    def chkpt_load(self):

        checkpoint_file = os.path.join(self.checkpoint_dir ,'checkpoint.pickle')

        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        episode = checkpoint_data['episode']
        timestep = checkpoint_data['timestep']
        cumulative_score = checkpoint_data['cumulative_score']
        
        if 'log_std' in checkpoint_data:
            self.log_std.assign(checkpoint_data['log_std'])

        print()
        #print(f"Checkpoint loaded from {checkpoint_file} episode : {episode} , log_std = {self.log_std}")
        print(f"Checkpoint loaded from {checkpoint_file} episode : {episode}")

        return episode, timestep, cumulative_score

    def prn(self):
        print()
        print(f'log_std is = {self.log_std}')
        print()


def receive_all(conn, size):
    """Helper function to receive all 'size' bytes."""
    data = b''
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise Exception("Connection closed or error while receiving data.")
        data += chunk 
    return data



def run():

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)


    encoder = EncodeState()
    
    agent = PPOAgent()
    agent.load()
    agent.prn()

    print("TESTING.....")

    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Server listening on {HOST}:{PORT}...")

        while True:

            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
    

            while True:


                data = receive_all(conn, 153620)

                observation = struct.unpack('38405f', data)

                observation = encoder.process(observation)
                observation = observation.numpy()

                action,_ = agent(observation,False)

                if observation is None:
                    print("loop broken due to no observation")
                    break
                                        

                response = struct.pack('2f', *action)
                conn.sendall(response)
            

            conn.close()


    except Exception as e:
        print(f"Error: {e}")

    finally:
        server_socket.close()
        print("Server socket closed.")

    print("Terminating the run.")
    sys.exit()



    

if __name__ == "__main__":

    try:
        run()

    except KeyboardInterrupt:
        sys.exit()
        
    finally:
        print("\nTerminating...")