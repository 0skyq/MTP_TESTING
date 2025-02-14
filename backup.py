import os
import sys
import glob
import math
import weakref
import pygame
import time
import random
import pickle
import math
import numpy as np
import logging
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
layers = tf.keras.layers
#tf.get_logger().setLevel('ERROR')



#VAR AUTO ENCODER PARAMETERS
IM_WIDTH = 160
IM_HEIGHT = 80
LATENT_DIM = 95

#ACTOR_CRITIC NETWORK PARAMETERS
OBSERVATION_DIM = 100
ACTION_DIM = 2


#HYPER PARAMETERS
ACTION_STD_INIT = 0.2
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
POLICY_CLIP = 0.2
GAMMA = 0.99
LAMBDA = 0.95
NO_OF_ITERATIONS = 15
SEED = 0

#TRAINING PARAMETS
TOTAL_TIMESTEPS = 2e6
EPISODE_LENGTH = 7500

#SIMULATION PARAMETERS 
TOWN = 'Town02'
CAR_NAME = 'model3'
NUMBER_OF_VEHICLES = 30
NUMBER_OF_PEDESTRIAN = 10
CONTINUOUS_ACTION = True
VISUAL_DISPLAY = False

VAR_AUTO_MODEL = 'VAE/'
VAR_AUTO_MODEL_PATH = 'VAE/var_auto_encoder_model'
DRL_MODEL_PATH = 'DRL'


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

    def __init__(self):
        self.host ="localhost"
        self.town = TOWN
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
            self.max_speed = 35.0
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
                        if x < 200:
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
                #print(f'steer = {self.previous_steer*0.9 + steer*0.1},throttle = {self.throttle*0.9 + throttle*0.1}')
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=self.throttle*0.9 + throttle*0.1))
                self.previous_steer = steer
                self.throttle = throttle
                #v = self.vehicle.get_velocity()
                #print(f'velocity = {np.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6}')
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

            #Update checkpoint for training
            # if not self.fresh_start:
            #     if self.checkpoint_frequency is not None:
            #         #print("ok_0")
            #         self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

            
            # Rewards are given below!
            done = False
            reward = 0

            if len(self.collision_history) != 0:
                #print("collison detected")
                done = True
                reward = -10
            elif self.distance_from_center > self.max_distance_from_center:
                #print("moved away from lane")
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
                        #print('less velocity')
                        reward = (self.velocity / self.min_speed) * centering_factor * angle_factor    
                    elif self.velocity > self.target_speed:
                        #print('more velocity')               
                        reward = (1.0 - (self.velocity-self.target_speed) / (self.max_speed-self.target_speed)) * centering_factor * angle_factor  
                    else:                       
                        #print('normal')                  
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





class EncodeState:
    def __init__(self):
        self.model_path = VAR_AUTO_MODEL_PATH
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.model.trainable = False  # Ensure model is in inference mode

        # Ensure BatchNorm layers do not update statistics
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.training = False  
        
        print()
        print(f'Variational AutoEncoder Loaded from {self.model_path}')
        print()

    def process(self, observation):
        image_obs = tf.convert_to_tensor(observation[0], dtype=tf.float32)
        image_obs = tf.expand_dims(image_obs, axis=0)  
        image_obs = self.model(image_obs, training=False)  # Ensure inference mode
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
        self.n_updates_per_iteration = NO_OF_ITERATIONS
        self.memory = Buffer()
        self.town = TOWN
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.models_dir = f'models'
        self.checkpoint_dir = f'checkpoints'


        #self.log_std = tf.Variable(tf.zeros(self.action_dim), trainable=True, dtype=tf.float32)
        self.log_std = tf.Variable(tf.fill((self.action_dim,), tf.math.log(self.action_std_init)), trainable=False, dtype=tf.float32)


        self.actor = Actor()
        self.critic = Critic()
        self.old_actor = Actor()
        self.old_critic = Critic()

        self.actor.compile(optimizer=self.optimizer)
        self.critic.compile(optimizer=self.optimizer)
        self.old_actor.compile(optimizer=self.optimizer)
        self.old_critic.compile(optimizer=self.optimizer)
        
        self.update_old_policy()

    
    def call(self, obs, train=True):

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
        
        return action.numpy().flatten()


    def update_old_policy(self):
        self.old_actor.set_weights(self.actor.get_weights())
        self.old_critic.set_weights(self.critic.get_weights())
    

    def get_action_and_log_prob(self, mean):
        std = tf.exp(self.log_std)  
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)
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
        return tf.convert_to_tensor(advantages, dtype=tf.float32), tf.convert_to_tensor(returns, dtype=tf.float32)


    def evaluate(self, obs, action):
        mean = self.actor(obs)

        if tf.reduce_any(tf.math.is_nan(mean)):
            print("NaN detected in the mean, exiting...")
            exit()

        
        std = tf.exp(self.log_std) 
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)

        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        values = self.critic(obs)

        return log_probs, values, entropy


    def learn(self):

        rewards = self.memory.rewards
        dones = self.memory.dones
        old_states = tf.squeeze(tf.stack(self.memory.observation, axis=0))
        old_actions = tf.squeeze(tf.stack(self.memory.actions, axis=0))
        old_logprobs = tf.squeeze(tf.stack(self.memory.log_probs, axis=0))

        values = self.critic(old_states)
        values = tf.squeeze(values)
        values = tf.concat([values, tf.zeros((1,))], axis=0) 

        advantages, returns = self.compute_advantages(rewards, values, dones)
        returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-7)
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-7)



        for i in range(self.n_updates_per_iteration):

            with tf.GradientTape(persistent=True) as tape:
                log_probs, values, dist_entropy = self.evaluate(old_states, old_actions)
                values = tf.squeeze(values)
                ratios = tf.exp(log_probs - old_logprobs)
                surr1 = ratios * advantages
                surr2 = tf.clip_by_value(ratios, 1 - self.clip, 1 + self.clip) * advantages
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - 0.01 * tf.reduce_mean(dist_entropy)
                critic_loss =  0.5 * self.loss(values, returns) 

            #grads_a = tape.gradient(actor_loss, self.actor.trainable_variables+ [self.log_std])
            grads_a = tape.gradient(actor_loss, self.actor.trainable_variables)
            grads_c = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads_a, self.actor.trainable_variables))
            #self.optimizer.apply_gradients(zip(grads_a, self.actor.trainable_variables+ [self.log_std]))
            self.optimizer.apply_gradients(zip(grads_c, self.critic.trainable_variables))
            
            #print(f"Entropy Loss: {tf.reduce_mean(dist_entropy).numpy()}")
            #print(f"Loss: {actor_loss.numpy() , critic_loss.numpy()}")



        self.update_old_policy()
        self.memory.clear()
        print("\nUPDATED THE WEIGHTS\n")

    def save(self):

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        print('... saving models ...')
        self.actor.save(self.models_dir + '/actor')
        self.critic.save(self.models_dir + '/critic')
       
 
        # Get all the weights and flatten them into a single list
        actor_weights = np.concatenate([weight.flatten() for weight in self.actor.get_weights()])
        critic_weights = np.concatenate([weight.flatten() for weight in self.critic.get_weights()])
        # Print first 10 values for the actor model
        print('Actor weights (first 10 values):')
        print(actor_weights[:10])
        # Print first 10 values for the critic model
        print('Critic weights (first 10 values):')
        print(critic_weights[:10])

       
        print(f"Model weights are saved at {self.models_dir}")


    def chkpt_save(self,episode,timestep,cumulative_score):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        checkpoint_file = os.path.join(self.checkpoint_dir ,'checkpoint.pickle')

        data = {
            'episode': episode,
            'timestep': timestep,
            'cumulative_score': cumulative_score,
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)


        print(f"Checkpoint saved as {checkpoint_file}")
        print()


    def load(self):

        print('... loading models ...')
        self.actor = tf.keras.models.load_model(self.models_dir + '/actor')
        self.critic = tf.keras.models.load_model(self.models_dir + '/critic')
        self.old_actor = tf.keras.models.load_model(self.models_dir + '/actor')
        self.old_critic = tf.keras.models.load_model(self.models_dir + '/critic')

        

        # Get all the weights and flatten them into a single list
        actor_weights = np.concatenate([weight.flatten() for weight in self.actor.get_weights()])
        critic_weights = np.concatenate([weight.flatten() for weight in self.critic.get_weights()])

        # Print first 10 values for the actor model
        print('Actor weights (first 10 values):')
        print(actor_weights[:10])

        # Print first 10 values for the critic model
        print('Critic weights (first 10 values):')
        print(critic_weights[:10])



        #self.update_old_policy()

        # Get all the weights and flatten them into a single list
        actor_weights = np.concatenate([weight.flatten() for weight in self.old_actor.get_weights()])
        critic_weights = np.concatenate([weight.flatten() for weight in self.old_critic.get_weights()])

        # Print first 10 values for the actor model
        print('Old Actor weights (first 10 values):')
        print(actor_weights[:10])

        # Print first 10 values for the critic model
        print('Old Critic weights (first 10 values):')
        print(critic_weights[:10])



        print(f"Model weights are loaded from {self.models_dir}")


    def chkpt_load(self):

        checkpoint_file = os.path.join(self.checkpoint_dir ,'checkpoint.pickle')

        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        episode = checkpoint_data['episode']
        timestep = checkpoint_data['timestep']
        cumulative_score = checkpoint_data['cumulative_score']
        
        print(f"Checkpoint loaded from {checkpoint_file}")
        
        return episode, timestep, cumulative_score




def train():

    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    CHECKPOINT_LOAD = True

    try:
        client, world = ClientConnection().setup()
        logging.info("CONNECTION HAS BEEN STEUP SUCCESSFULLY.")
        print("CONNECTION HAS BEEN STEUP SUCCESSFULLY.")
        print()
    except:
        logging.error("CONNECTION HAS BEEN REFUSED BY THE SERVER.")
        ConnectionRefusedError
        print("CONNECTION HAS BEEN REFUSED BY THE SERVER.")
        print()
   

    env = CarlaEnvironment(client, world,TOWN)
    encoder = EncodeState()
    
    if CHECKPOINT_LOAD:
        print()
        print('LOADING FROM CHECKPOINT....')
        print()
        agent = PPOAgent()
        episode , timestep , cumulative_score = agent.chkpt_load()
        agent.load()
    else:

        agent = PPOAgent()



    while timestep < TOTAL_TIMESTEPS:
        #print()
        #print('training....')
        observation = env.reset()
        observation = encoder.process(observation)

        current_ep_reward = 0
        t1 = datetime.now()

        for t in range(EPISODE_LENGTH): 

            observation = observation.numpy()

            action = agent(observation)
            
            #print(f'actions are {action}')

            observation, reward, done, info = env.step(action)


            if observation is None:
                break

            observation = encoder.process(observation)

            agent.memory.rewards.append(reward)
            agent.memory.dones.append(done)

            timestep +=1
            current_ep_reward += reward


            if done:
                episode += 1
                t2 = datetime.now()
                t3 = t2-t1
                episodic_length.append(abs(t3.total_seconds()))
                break

        deviation_from_center += info[1]
        distance_covered += info[0]
        
        scores.append(current_ep_reward)

        if CHECKPOINT_LOAD:
            cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
        else:
            cumulative_score = np.mean(scores)
        

        print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score),', Distance Covered:{}'.format(info[0]))

        if episode % 10 == 0:
            agent.learn()
            agent.chkpt_save(episode,timestep,cumulative_score)

        if episode%100 == 0:
            agent.save()
            agent.chkpt_save(episode,timestep,cumulative_score)
 

    sys.exit()


if __name__ == "__main__":
    try:
        train()
        #test()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nTerminating...")