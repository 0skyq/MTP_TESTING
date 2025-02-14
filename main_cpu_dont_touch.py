import os
import sys
import glob
import time
import random
import pickle
import csv
import cv2
import logging
import math
import weakref
import pygame
import numpy as np
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter


# import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# import tensorflow_probability as tfp
# tfd = tfp.distributions
# layers = tf.keras.layers
# ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


#CODE PARAMETERS
TRAIN = True
TOWN = 'Town02'
CHECKPOINT_LOAD = False
SEED = 0
RUN_NAME ='PPO'

#AUTO ENCODER PARAMETERS
IM_WIDTH = 160
IM_HEIGHT = 80
LATENT_DIM = 95

#DRL NETWORK PARAMETERS
MEMORY_SIZE = 10000
EPISODES = 1000
EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 3e6
TEST_TIMESTEPS = 5e4
PPO_CHECKPOINT_DIR = 'preTrained_models/PPO/'
VAR_AUTO_MODEL = 'VAE/'
OBS_DIM = 100
ACTION_DIM = 2
MODEL_NUMBER = 9

VAR_AUTO_MODEL_PATH = 'VAE/var_auto_encoder_model'

#HYPER PARAMETERS
ACTION_STD_INIT = 0.2
PPO_LEARNING_RATE= 1e-4
BATCH_SIIZE = 4
POLICY_CLIP = 0.2
GAMMA = 0.99
NO_OF_ITERATIONS = 7

#SIMULATION PARAMETERS 
CAR_NAME = 'model3'
NUMBER_OF_VEHICLES = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = False

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

print()
print(f'DEVICE USING : {device}')
print()

try:
    sys.path.append(glob.glob('./carla/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('Couldn\'t import Carla egg properly')

import carla

class ClientConnection:

    def __init__(self,town):
        self.host ="localhost"
        self.town = town
        self.client = None
        self.port = 2000
        self.timeout= 20.0

    def setup(self):
        try:

            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.load_world(self.town)
            self.world.set_weather(carla.WeatherParameters.CloudyNoon)

            return self.client, self.world

        except Exception as e:
            print('Failed to make a connection with the server: {}'.format(e))

            if self.client.get_client_version != self.client.get_server_version:
                print("There is a Client and Server version mismatch! Please install or download the right versions.")



class EncodeState():

    def __init__(self, latent_dim):

        self.latent_dim = latent_dim
        #self.device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        try:
            #self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
            self.conv_encoder = VariationalEncoder(self.latent_dim).to(device)
            self.conv_encoder.load()
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False

        except Exception as e:
            print(f'Encoder could not be initialized {e}')
            sys.exit()
    
    def process(self, observation):

        #image_obs = torch.tensor(observation[0], dtype=torch.float).to(self.device)
        image_obs = image_obs = torch.tensor(observation[0], dtype=torch.float).to(device)
        image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.permute(0,3,2,1)
        image_obs = self.conv_encoder(image_obs)
        #navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(device)
        observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
        
        return observation

class VariationalEncoder(nn.Module):
    
    def __init__(self, latent_dims):  

        super(VariationalEncoder, self).__init__()

        self.model_file = os.path.join(VAR_AUTO_MODEL, 'var_encoder_model.pth')

        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # 79, 39
            nn.LeakyReLU())

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 40, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2),  # 19, 9
            nn.LeakyReLU())

        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2),  # 9, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.linear = nn.Sequential(
            nn.Linear(9*4*256, 1024),
            nn.LeakyReLU())

        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu =  self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file,map_location=device))
        print(f"AUTOENCODER MODEL LOADED FROM {self.model_file}")

     



# class EncodeState():
#     def __init__(self,latent_dim):
#         self.model_path = VAR_AUTO_MODEL_PATH
#         self.model = tf.keras.models.load_model(self.model_path, compile=False)
#         self.model.trainable = False  

#         for layer in self.model.layers:
#             if isinstance(layer, tf.keras.layers.BatchNormalization):
#                 layer.training = False  

#         print(f'Variational AutoEncoder Loaded from {self.model_path}')

#     def process(self, observation):
#         image_obs = tf.convert_to_tensor(observation[0], dtype=tf.float32)
#         image_obs = tf.expand_dims(image_obs, axis=0)  
#         image_obs = self.model(image_obs, training=False)  

#         navigation_obs = tf.convert_to_tensor(observation[1], dtype=tf.float32)
#         observation = tf.concat([tf.reshape(image_obs, [-1]), navigation_obs], axis=-1)

#         return observation




class CarlaEnvironment():

    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True) -> None:


        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.action_space = self.get_discrete_action_space()
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start=True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.create_pedestrians()

    def reset(self):

        try:
            
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()


            # Blueprint of our main vehicle
            vehicle_bp = self.get_vehicle(CAR_NAME)

            if self.town == "Town07":
                transform = self.map.get_spawn_points()[20] #Town7  is 38 
                self.total_distance = 750
            elif self.town == "Town02":
                transform = self.map.get_spawn_points()[30] #Town2 is 30
                self.total_distance = 500
            else:
                transform = self.map.get_spawn_points()[12] #40 nocd 
                self.total_distance = 500

            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)


            # Camera Sensor
            self.camera_obj = CameraSensor(self.vehicle)
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            # Third person view of our vehicle in the Simulated env
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.distance_traveled = 0.0
            self.center_lane_deviation = 0.0
            self.target_speed = 22 #km/h
            self.max_speed = 25.0
            self.min_speed = 15.0
            self.max_distance_from_center = 3
            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.center_lane_deviation = 0.0
            self.distance_covered = 0.0


            if self.fresh_start:
                #print("fresh start")

                self.current_waypoint_index = 0
                # Waypoint nearby angle and distance from it
                self.route_waypoints = list()
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)

                for x in range(self.total_distance):
                #for x in range(1000):
                    if self.town == "Town07":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02": #200
                        if x < 50:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]

                        #print(f"x={x} next_waypoint = {next_waypoint}")
                       
                    else:
                        if x < 300:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]

                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint

            else:
                #print("waypoint update")
                # Teleport vehicle to last checkpoint
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                transform = waypoint.transform
                self.vehicle.set_transform(transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index

            self.navigation_obs = np.array([self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])

                        
            time.sleep(0.5)
            self.collision_history.clear()

            self.episode_start_time = time.time()
            return [self.image_obs, self.navigation_obs]

        except Exception as e:
            print(f"Error while Resetting : {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    def step(self, action_idx):
        try:

            self.timesteps+=1
            self.fresh_start = False

            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            # Action fron action space for contolling the vehicle with a discrete action
            if self.continous_action_space:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = float((action_idx[1] + 1.0)/2)
                throttle = max(min(throttle, 1.0), 0.0)
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=self.throttle*0.9 + throttle*0.1))
                self.previous_steer = steer
                self.throttle = throttle
            else:
                steer = self.action_space[action_idx]
                if self.velocity < 20.0:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
                self.previous_steer = steer
                self.throttle = 1.0
            
            # Traffic Light state
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)

            self.collision_history = self.collision_obj.collision_data            

            # Rotation of the vehicle in correlation to the map/lane
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            self.location = self.vehicle.get_location()


            #transform = self.vehicle.get_transform()
            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]

            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_diff(fwd, wp_fwd)

            # #Update checkpoint for training
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    #print("ok_0")
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            
            # Rewards are given below!
            done = False
            reward = 0

            if len(self.collision_history) != 0:
                #print("collison detected")
                done = True
                reward = -10
            elif self.distance_from_center > self.max_distance_from_center:
                #print("moved away from lane")
                # print(self.distance_from_center)
                # print(self.current_waypoint)
                # print(self.next_waypoint)
                done = True
                reward = -10
            elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
                #print("less than min velocity")
                reward = -10
                done = True
            elif self.velocity > self.max_speed:
                #print("exceeded max velocity")
                reward = -10
                done = True

            # Interpolated from 1 when centered to 0 when 3 m from center
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 30 degress of road
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

            if not done:
                if self.continous_action_space:
                    if self.velocity < self.min_speed:
                        reward = (self.velocity / self.min_speed) * centering_factor * angle_factor    
                    elif self.velocity > self.target_speed:               
                        reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                    else:                                         
                        reward = 1.0 * centering_factor * angle_factor 
                else:
                    reward = 1.0 * centering_factor * angle_factor

            if self.timesteps >= 2e6:
                #print("ok_1")
                done = True
            elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
                print("Reached destination -- Repeat")
                done = True
                self.fresh_start = True
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                        #print("ok_3")
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0
                        #print("ok_4")



            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)

            self.image_obs = self.camera_obj.front_camera.pop(-1)
            normalized_velocity = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])
            
            # Remove everything that has been spawned in the env
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)
                
                for sensor in self.sensor_list:
                    sensor.destroy()
                
                self.remove_sensors()
                
                for actor in self.actor_list:
                    actor.destroy()
            
            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation]

        except Exception as e:
            print(f"Error while step  : {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()

    def create_pedestrians(self):
        try:

            # Our code for this method has been broken into 3 sections.

            # 1. Getting the available spawn points in  our world.
            # Random Spawn locations for the walker
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            # 2. We spawn the walker actor and ai controller
            # Also set their respective attributes
            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(
                    self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find(
                    'controller.ai.walker')
                # Walkers are made visible in the simulation
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # They're all walking not running on their recommended speed
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute(
                        'speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            # set how many pedestrians can cross the road
            #self.world.set_pedestrians_cross_factor(0.0)
            # 3. Starting the motion of our pedestrians
            for i in range(0, len(self.walker_list), 2):
                # start walker
                all_actors[i].start()
            # set walk to random point
                all_actors[i].go_to_location(
                    self.world.get_random_location_from_navigation())

        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.walker_list])

    def set_other_vehicles(self):
        try:
            # NPC vehicles generated and set to autopilot
            # One simple for loop for creating x number of vehicles and spawing them into the world
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(
                    bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.actor_list.append(other_vehicle)
            print("NPC vehicles have been generated in autopilot mode.")
        except:
            self.client.apply_batch(
                [carla.command.DestroyActor(x) for x in self.actor_list])

    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)

    def get_world(self) -> object:
        return self.world

    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()

    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle

    def distance_to_line(self, A, B, p):
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom

    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

    def get_discrete_action_space(self):
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint

    def set_vehicle(self, vehicle_bp, spawn_points):
        # Main vehicle spawned into the env
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None

class CameraSensor():

    def __init__(self, vehicle):
        self.sensor_name = 'sensor.camera.semantic_segmentation'
        self.parent = vehicle
        self.front_camera = list()
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(weak_self, image))

    # Main front camera is setup and provide the visual observations for our network.
    def _set_camera_sensor(self, world):
        front_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        front_camera_bp.set_attribute('image_size_x', f'160')
        front_camera_bp.set_attribute('image_size_y', f'80')
        front_camera_bp.set_attribute('fov', f'125')
        front_camera = world.spawn_actor(front_camera_bp, carla.Transform(
            carla.Location(x=2.4, z=1.5), carla.Rotation(pitch= -10)), attach_to=self.parent)
        return front_camera

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.CityScapesPalette)
        placeholder = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = placeholder.reshape((image.width, image.height, 4))
        target = placeholder1[:, :, :3]
        self.front_camera.append(target)#/255.0)

class CameraSensorEnv:

    def __init__(self, vehicle):

        pygame.init()
        self.display = pygame.display.set_mode((720, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.sensor_name = 'sensor.camera.rgb'
        self.parent = vehicle
        self.surface = None
        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensorEnv._get_third_person_camera(weak_self, image))

    # Third camera is setup and provide the visual observations for our environment.

    def _set_camera_sensor(self, world):

        thrid_person_camera_bp = world.get_blueprint_library().find(self.sensor_name)
        thrid_person_camera_bp.set_attribute('image_size_x', f'720')
        thrid_person_camera_bp.set_attribute('image_size_y', f'720')
        third_camera = world.spawn_actor(thrid_person_camera_bp, carla.Transform(
            carla.Location(x=-4.0, z=2.0), carla.Rotation(pitch=-12.0)), attach_to=self.parent)
        return third_camera

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        placeholder1 = array.reshape((image.width, image.height, 4))
        placeholder2 = placeholder1[:, :, :3]
        placeholder2 = placeholder2[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(placeholder2.swapaxes(0, 1))
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()

class CollisionSensor:

    def __init__(self, vehicle) -> None:
        self.sensor_name = 'sensor.other.collision'
        self.parent = vehicle
        self.collision_data = list()
        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    # Collision sensor to detect collisions occured in the driving process.
    def _set_collision_sensor(self, world) -> object:
        collision_sensor_bp = world.get_blueprint_library().find(self.sensor_name)
        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, z=0.5))
        collision_sensor = world.spawn_actor(
            collision_sensor_bp, sensor_relative_transform, attach_to=self.parent)
        return collision_sensor

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_data.append(intensity)

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







class ActorCritic(nn.Module):

    def __init__(self, obs_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        #self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #self.device = torch.device('cpu')

        self.cov_var = torch.full((self.action_dim,), action_std_init, device=device)
        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)

        self.actor = nn.Sequential(
                        nn.Linear(self.obs_dim, 500),
                        nn.Tanh(),
                        nn.Linear(500, 300),
                        nn.Tanh(),
                        nn.Linear(300, 100),
                        nn.Tanh(),
                        nn.Linear(100, self.action_dim),
                        nn.Tanh()
                    )
        
        self.critic = nn.Sequential(
                        nn.Linear(self.obs_dim, 500),
                        nn.Tanh(),
                        nn.Linear(500, 300),
                        nn.Tanh(),
                        nn.Linear(300, 100),
                        nn.Tanh(),
                        nn.Linear(100, 1)
                    )

        self.actor.to(device)
        self.critic.to(device)


    def forward(self):
        raise NotImplementedError
    
    def normalize(self,obs):
        
        obs = torch.clamp(obs, min=-1e3, max=1e3)

        # max_val = obs.max()
        # min_val = obs.min()
        # obs = 2 * (obs - min_val) / (max_val - min_val) - 1


        return obs
    
    def set_action_std(self, new_action_std):
        self.cov_var = torch.full((self.action_dim,), new_action_std)

    def get_value(self, obs):

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        obs = self.normalize(obs)

        return self.critic(obs)
    
    def get_action_and_log_prob(self, obs):

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        #print(obs)


        obs  = self.normalize(obs)

        mean = self.actor(obs)

        #print(f"mean value is {mean}")

        #print('ok_4')

        if torch.isnan(mean).any():
            print("Nan values found in mean")
            print(mean)

        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)


        #print(f"action : {action.detach().squeeze()} , logprob : {log_prob.detach().squeeze()}")

        return action.detach().squeeze(), log_prob.detach().squeeze(),mean.detach().squeeze()

    
    def evaluate(self, obs, action):

        #print("started evaluation")
        obs = self.normalize(obs)

        mean = self.actor(obs)

        if torch.isnan(mean).any():
            print("Nan values found in mean")
            print(mean)
        
        cov_var = self.cov_var.expand_as(mean)
        cov_mat = torch.diag_embed(cov_var)

        dist = MultivariateNormal(mean, cov_mat)

        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(obs)

        #print("ended evaluation")

        return logprobs, values, dist_entropy

    def clear(self):
        print("MEMORY CLEARED ")
        del self.observation[:]    
        del self.actions[:]        
        del self.log_probs[:]      
        del self.rewards[:]
        del self.dones[:]




class PPOAgent(object):

    def __init__(self, town, action_std_init):
        
        self.obs_dim = 100
        self.action_dim = 2
        self.clip = POLICY_CLIP
        self.gamma = GAMMA
        self.n_updates_per_iteration = NO_OF_ITERATIONS
        self.lr = PPO_LEARNING_RATE
        self.action_std = action_std_init
        #self.encode = EncodeState(LATENT_DIM)
        self.memory = Buffer()
        self.town = town

        #self.device = torch.device('cpu')

        self.checkpoint_file_no = 0
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr}])

        self.old_policy = ActorCritic(self.obs_dim, self.action_dim, self.action_std)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def get_action(self, obs, train):

        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)

            action, logprob , mean= self.old_policy.get_action_and_log_prob(obs.to(device))
            
        if train:
            self.memory.observation.append(obs.to(device))
            self.memory.actions.append(action)
            self.memory.log_probs.append(logprob)

        #return action.detach().cpu().numpy().flatten()
        #return action.detach().numpy().flatten()
        return action.detach().cpu().numpy().flatten(),mean.detach().numpy()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)
        return self.action_std

    def learn(self):
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(self.device)
        # old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(self.device)
        # old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(self.device)

        old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)

        for i in range(self.n_updates_per_iteration):

            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            values = torch.squeeze(values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(values, rewards) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            print()
            #print(f"Iteration {i+1}/{self.n_updates_per_iteration}")
            #print(f"Entropy: {dist_entropy.mean().item()}")
            print(f"Total Loss: {loss.mean().item()}")
            #print(f"Advantages: {advantages.mean().item()}")

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()
        print()
        print("UPDATED THE WEIGHTS")

        print()

    def save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)
        print()
        print(f"MODEL SAVED AT {checkpoint_file}")
        print()

    def chkpt_save(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2])
        if self.checkpoint_file_no !=0:
            self.checkpoint_file_no -=1
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        torch.save(self.old_policy.state_dict(), checkpoint_file)
        print()
        print(f"CHECKPOINT SAVED AT {checkpoint_file}")
        print()
   
    def load(self):
        self.checkpoint_file_no = len(next(os.walk(PPO_CHECKPOINT_DIR+self.town))[2]) - 1
        checkpoint_file = PPO_CHECKPOINT_DIR+self.town+"/ppo_policy_" + str(self.checkpoint_file_no)+"_.pth"
        self.old_policy.load_state_dict(torch.load(checkpoint_file,map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_file,map_location=device))
        print()
        print(f"DRL MODEL LOADED FROM  {checkpoint_file}")
        print()



def runner():

    #args = parse_args()
    train = TRAIN
    town = TOWN
    checkpoint_load = CHECKPOINT_LOAD
    total_timesteps = TOTAL_TIMESTEPS
    action_std_init = ACTION_STD_INIT
    run_name = RUN_NAME

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    #tf.random.set_seed(SEED)
    #np.random.seed(SEED)
    #random.seed(SEED)

    action_std_decay_rate = 0.05
    min_action_std = 0.05   
    action_std_decay_freq = 5e6
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    print()
    print(f'TRAIN : {train}')
    print(f'TOWN  : {town}')
    print(f'CHECKPOINT LOAD : {checkpoint_load}')
    print(f'TOTAL_TIMESTEPS : {total_timesteps}')
    print(f'ACTION_STD_INIT : {action_std_init}')
    #print(f'Device : {device}')
    print()
    
    
    if train == True:
        writer = SummaryWriter(f"runs/{town}/{run_name}_{action_std_init}")
    else:
        writer = SummaryWriter(f"runs/{town}/{run_name}_{action_std_init}_TEST")

    # writer.add_text("hyperparameters","|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))


    try:
        client, world = ClientConnection(town).setup()
        logging.info("CARLA CONNECTION HAS BEEN STEUP SUCCESSFULLY.")
        print("CARLA CONNECTION HAS BEEN STEUP SUCCESSFULLY.")
        print()
    except:
        logging.error("CARLA CONNECTION HAS BEEN REFUSED BY THE SERVER.")
        ConnectionRefusedError
        print("CARLA CONNECTION HAS BEEN REFUSED BY THE SERVER.")
        print()
   
   
    if train:
        env = CarlaEnvironment(client, world,town)
    else:
        env = CarlaEnvironment(client, world,town, checkpoint_frequency=None)
    
    encode = EncodeState(LATENT_DIM)

    try:
        time.sleep(0.5)
        
        if checkpoint_load:
            chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2]) - 1
            chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']
                timestep = data['timestep']
                cumulative_score = data['cumulative_score']
                action_std_init = data['action_std_init']
            agent = PPOAgent(town, action_std_init)
            agent.load()
        else:
            
            if train == False:
                agent = PPOAgent(town, action_std_init)
                agent.load()
                for params in agent.old_policy.actor.parameters():
                    params.requires_grad = False
            else:
                agent = PPOAgent(town, action_std_init)
     
     
        if train:
            #Training
            while timestep < total_timesteps:
                #print("training...")
                observation = env.reset()
                observation = encode.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(EPISODE_LENGTH):

                    # select action with policy
                    observation = observation.numpy()
                    # print(f"Obseavtion is : ")
                    # print(observation)

                    action ,mean = agent.get_action(observation, train=TRAIN)

                    # print(f"actions are :")
                    # print(action)
                    observation, reward, done, info = env.step(action)

                    if observation is None:
                        print("Observation is none\n")
                        break

                    observation = encode.process(observation)

                    agent.memory.rewards.append(reward)
                    agent.memory.dones.append(done)
                    
                    timestep +=1
                    current_ep_reward += reward
                    
                    if timestep % action_std_decay_freq == 0:
                        action_std_init =  agent.decay_action_std(action_std_decay_rate, min_action_std)

                    if timestep == total_timesteps -1:
                        agent.chkpt_save()

                    # break; if the episode is over
                    if done:
                        episode += 1
                        t2 = datetime.now()
                        t3 = t2-t1
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                
                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                
                if checkpoint_load:
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                else:
                    cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score),', Distance Covered:{}'.format(info[0]))
             
                if episode % 10 == 0:
                    agent.learn()
                    agent.chkpt_save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    if chkt_file_nums != 0:
                        chkt_file_nums -=1
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                    
                if episode % 5 == 0:
                    
                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    writer.add_scalar("Average Deviation from Center/episode", deviation_from_center/5, episode)
                    writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center/5, timestep)
                    writer.add_scalar("Average Distance Covered (m)/episode", distance_covered/5, episode)
                    writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered/5, timestep)

                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0

                if episode % 100 == 0:
                    agent.save()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/PPO/{town}'))[2])
                    chkpt_file = f'checkpoints/PPO/{town}/checkpoint_ppo_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                        
            print("Terminating the run.")
            sys.exit()

        else:
            print("testing..")

            folder_count = 1
            while timestep < TEST_TIMESTEPS:

                save_dir = f"captured_images/Episode_images_{folder_count}"
                
                if os.path.exists(save_dir):
                    print(f"Folder already exists: {save_dir}")
                    break 

                print(f"saving in {save_dir}")

                os.makedirs(save_dir, exist_ok=True)
                frame_count = 0

                data_to_append = ['sr.no', 'throttle','velocity', 'norm_velocity', 'nor_dis_center', 'nor_angle','mean[0]','mean[1]','reward','exe_time']


                with open(f"captured_images/Episode_data_{folder_count}.csv",mode = 'a',newline = '') as file:
                    writer = csv.writer(file)
                    
                    writer.writerow(data_to_append)


                observation = env.reset()
                #observation = encode.process(observation_view)

                #print("ok1")

                current_ep_reward = 0
                t1 = datetime.now()

                #KEY CODE {obs -> action -> obs,reward -> obs}
                for t in range(EPISODE_LENGTH):
                    
                    data_to_append = []

                    image_array = np.array(observation[0], dtype=np.uint8)
                    save_path = os.path.join(save_dir, f"frame_{frame_count}.png")
                    cv2.imwrite(save_path, image_array)

                    data_to_append.append(f'frame_{frame_count}')

                    for x in observation[1]:
                        data_to_append.append(x)

                    s_time = time.time()

                    observation = encode.process(observation)

                    action , mean = agent.get_action(observation, train=False)

                    e_time = time.time()

                    observation, reward, done, info = env.step(action)


                    frame_count +=1
                    for x in mean:
                        data_to_append.append(x)

                    data_to_append.append(reward)


                    if observation is None:
                        break

                    exe_time = (e_time-s_time)


                    data_to_append.append(exe_time)

                    with open(f"captured_images/Episode_data_{folder_count}.csv",mode = 'a',newline = '') as file:
                        writer = csv.writer(file)
                        
                        writer.writerow(data_to_append)


                    timestep +=1
                    current_ep_reward += reward
                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break

                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score),', Distance Covered:{}'.format(info[0]))
                
                # writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                # writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                # writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                # writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                # writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                # writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                # writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                # writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                # writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0
                folder_count+=1

                if folder_count>10:
                    break

            print("start over")    
            print("Terminating the run.")
            sys.exit()

    finally:
        sys.exit()


if __name__ == "__main__":
    try:        
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')

