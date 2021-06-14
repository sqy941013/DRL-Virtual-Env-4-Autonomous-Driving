import gym
from gym import spaces
import numpy as np
# from os import path
import sim_client
import numpy as np
import copy
import collections as col
import os
import time
import sys


class TorcsEnv:
    terminal_judge_start = 50  # 1000  # 如果超过一定步数没有进展，则终止该回合的训练
    termination_limit_progress = 0.5  # [km/h], 如果车辆该步的进展小于该阈值，则终止该回合的训练
    default_speed = 50  # 默认速度

    initial_reset = True

    def __init__(self, throttle=False, gear_change=False):
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        self.client = sim_client.Client()

        ##print("launch torcs")
        # TODO：加入开启模拟环境的脚本
        # os.system('sh autostart.sh')
        time.sleep(0.5)

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
        low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
        self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
        # print("Step")
        client = self.client

        this_action = self.agent_to_torcs(u)

        action_torcs = client.R.d

        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        if self.throttle is False:
            sys.exit()

            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer'] * 50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1 / (client.S.d['speedX'] + .1)
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  换挡由模拟环境内的车辆控制器控制

        obs_pre = copy.deepcopy(client.S.d)

        client.respond_to_server()

        client.get_servers_input()

        obs = client.S.d

        self.observation = self.make_observaton(obs)

        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        damage = np.array(obs['damage'])
        rpm = np.array(obs['rpm'])

        # 奖励函数 progress
        # TODO：改进策略，引入自适应奖励函数
        # 思想为，奖励函数过于苛刻会导致前期训练过慢，或算法难以收敛，因此需要将前期奖励放宽，逐渐严格
        # progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])
        progress = sp * np.cos(obs['angle']) - np.abs(sp * np.sin(obs['angle'])) - sp * np.abs(
            obs['trackPos'])

        reward = progress

        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -100
            episode_terminate = True
            client.R.d['meta'] = True

        episode_terminate = False

        # ---------------------------------------------------
        # TODO:
        '''
        if self.terminal_judge_start < self.time_step:  # Episode terminates if the progress of agent is small
            if abs(progress) < self.termination_limit_progress:
                print("No progress", progress)
                reward = -100  # KAUSHIK ADDED THIS
                episode_terminate = True
                client.R.d['meta'] = True
        # ---------------------------------------------------
        '''

        if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True

        if client.R.d['meta'] is True:  # Send a reset signal
            self.initial_run = False
            client.R.d['steer'] = 0
            client.R.d['accel'] = 0
            client.R.d['brake'] = 0
            client.respond_to_server()
            time.sleep(2)
            # client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        # print("Reset")
        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

        self.client.close_connection()
        # self.client.reset_env()
        self.client = sim_client.Client()
        self.time_step = 0
        self.reset_torcs()

        self.client.MAX_STEPS = np.inf

        self.client.get_servers_input()  # Get the initial input from torcs

        obs = self.client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False

        time.sleep(2)

        return self.get_obs()

    def end(self):
        # TODO：关闭虚拟环境的程序
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        # print("relaunch torcs")
        # TODO：reset 虚拟环境
        '''
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)
        '''

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        return torcs_action

    def make_observaton(self, raw_obs):
        names = ['speedX', 'speedY', 'speedZ', 'angle', 'damage',
                 'rpm',
                 'track',
                 'trackPos']
        Observation = col.namedtuple('Observaion', names)

        return Observation(speedX=np.array(raw_obs['speedX'], dtype=np.float32) / 300.0,
                           speedY=np.array(raw_obs['speedY'], dtype=np.float32) / 300.0,
                           speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / 300.0,
                           angle=np.array(raw_obs['angle'], dtype=np.float32) / 3.1416,
                           damage=np.array(raw_obs['damage'], dtype=np.float32),
                           rpm=np.array(raw_obs['rpm'], dtype=np.float32) / 10000,
                           track=np.array(raw_obs['track'], dtype=np.float32) / 200.,
                           trackPos=np.array(raw_obs['trackPos'], dtype=np.float32) / 1.)

        '''
        return Observation(speedX=np.array(raw_obs['speedX'], dtype=np.float32) / 300.0,
                           speedY=np.array(raw_obs['speedY'], dtype=np.float32) / 300.0,
                           speedZ=np.array(raw_obs['speedZ'], dtype=np.float32) / 300.0,
                           angle=np.array(raw_obs['angle'], dtype=np.float32) / 3.1416,
                           damage=np.array(raw_obs['damage'], dtype=np.float32),
                           rpm=np.array(raw_obs['rpm'], dtype=np.float32) / 10000,
                           trackPos=np.array(raw_obs['trackPos'], dtype=np.float32) / 1.)
        '''
