#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import math
from enum import Enum
import numpy as np
from scipy.spatial import KDTree


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 0.5  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yawrate = 15.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.4  # [m/ss]
        self.max_dyawrate = 200.0 * math.pi / 180.0  # [rad/ss]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.v_reso = self.max_accel * self.dt / 10.0  # [m/s]
        self.yawrate_reso = self.max_dyawrate * self.dt / 10.0  # [rad/s]
        self.predict_time = 10  # [s]
        self.to_goal_cost_gain = 5
        self.speed_cost_gain = 0.1
        self.obstacle_cost_gain = 0.0
        self.dist_cost_gain = 0.000
        self.robot_type = RobotType.rectangle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 0.4  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.4  # [m] for collision check
        self.robot_length = 0.6  # [m] for collision check

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


class DWA:
    def __init__(self, config):
        self.config = config
        pass

    def plan(self, state, goal, ob):
        """
        Dynamic Window Approach control
        :param state: state [x, y, yaw, v, w]
        :param goal: (x, y) in robot frame
        :param ob: obstacle points (x, y)
        :return: (bool)
        """
        obs_tree = KDTree(ob)
        v_max = min(state[3] + self.config.max_accel * self.config.dt, self.config.max_speed)
        v_min = max(state[3] - self.config.max_accel * self.config.dt, self.config.min_speed)
        w_max = min(state[4] + self.config.max_dyawrate * self.config.dt, self.config.max_yawrate)
        w_min = max(state[4] - self.config.max_dyawrate * self.config.dt, -self.config.max_yawrate)

        min_cost = np.inf
        min_to_goal_cost = np.inf
        min_speed_cost = np.inf
        min_obs_cost = np.inf
        min_dist_cost = np.inf

        best_traj = []
        best_state = []
        for v in np.arange(v_min, v_max, self.config.v_reso):
            # print("v:", v)
            for w in np.arange(w_min, w_max, self.config.yawrate_reso):
                traj = self.motion_predict(v, w)

                to_goal_cost = self.calc_to_goal_cost(traj, goal)
                speed_cost = self.calc_speed_cost(traj, v_max)
                obs_cost = self.calc_obs_cost(traj, obs_tree)
                dist_cost = self.calc_dist_cost(traj, goal)

                cost = self.config.to_goal_cost_gain * to_goal_cost + \
                       self.config.speed_cost_gain * speed_cost + \
                       self.config.obstacle_cost_gain * obs_cost + \
                       self.config.dist_cost_gain * dist_cost

                if cost < min_cost:
                    min_cost = cost
                    min_to_goal_cost = self.config.to_goal_cost_gain * to_goal_cost
                    min_speed_cost = self.config.speed_cost_gain * speed_cost
                    min_obs_cost = self.config.obstacle_cost_gain * obs_cost
                    min_dist_cost = self.config.dist_cost_gain * dist_cost
                    best_state = [0, 0, 0, v, w]
                    best_traj = traj

        # print("cost: ", min_cost)
        # print("goal cost: ", min_to_goal_cost)
        # print("speed cost: ", min_speed_cost)
        # print("obs cost: ", min_obs_cost)
        # print("dist cost: ", min_dist_cost)

        return best_state, best_traj

    def motion_predict(self, v, w):
        new_state = [0, 0, 0, v, w]
        traj = []
        for t in np.arange(0, self.config.predict_time, self.config.dt):
            new_state[2] += w * self.config.dt
            # while new_state[2] > np.pi:
            #     new_state -= 2 * np.pi
            # while new_state[2] < -np.pi:
            #     new_state += 2 * np.pi
            # if w > abs(0.001):
            #     new_state[0] += v / w * (np.sin(new_state[2]) - np.sin(new_state[2] - w * self.config.dt))
            #     new_state[1] += v / w * (-np.cos(new_state[2]) + np.cos(new_state[2] - w * self.config.dt))
            # else:
            new_state[0] += v * np.cos(new_state[2]) * self.config.dt
            new_state[1] += v * np.sin(new_state[2]) * self.config.dt
            traj.append([new_state[0], new_state[1], new_state[2], v, w])
        # print(len(traj))
        return traj

    def calc_to_goal_cost(self, traj, goal):
        """
        delta yaw angle
        """
        target_angle = np.arctan2(goal[1], goal[0])
        # print(target_angle, traj[-1][2])
        return abs(traj[-1][2] - target_angle) / (np.pi / 1)

    def calc_speed_cost(self, traj, v_max):
        """
        delta v between current to max
        """
        return abs(v_max - traj[-1][3]) / v_max

    def calc_obs_cost(self, traj, obs_tree):
        """
        distance to nearest obstacle
        """
        distance, index = obs_tree.query(np.array([traj[-1][0], traj[-1][1]]))
        if distance > 2:
            distance = 2
        return (2 - distance) / 2

    def calc_dist_cost(self, traj, goal):
        """
        distance to goal
        """
        predict_dist = np.sqrt(np.square(goal[0] - traj[-1][0]) + np.square(goal[1] - traj[-1][1]))
        current_dist = np.sqrt(np.square(goal[0] - traj[0][0]) + np.square(goal[1] - traj[0][1]))
        return predict_dist / current_dist
