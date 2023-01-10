#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from enum import Enum

import numpy as np
from scipy.spatial import KDTree
import time

class DWA:
    def __init__(self):
        self.max_speed = 0.5  # [m/s]
        self.min_speed = 0  # [m/s]
        self.max_yawrate = 90 * math.pi / 180.0  # [rad/s]
        self.max_accel = 1  # [m/ss]
        self.max_dyawrate = 20.0 * math.pi / 180.0  # [rad/ss]
        self.dt = 0.07  # [s] Time tick for motion prediction
        self.v_reso = self.max_accel*self.dt/10.0  # [m/s]
        self.yawrate_reso = self.max_dyawrate*self.dt/10.0  # [rad/s]
        self.predict_time = 2  # [s]
        self.to_goal_cost_gain = 0.6
        self.speed_cost_gain = 0.5  # 0.1
        self.obstacle_cost_gain = 2 # 1.5
        self.vstep = 5
        self.wstep = 10
        self.robot_size = 0.2
        self.threshold = 0.2

    def window(self, v, w):
        """
        根据当前v。w计算速度窗口
        :param v: 机器人坐标系下的当前速度
        :param w: 机器人坐标系下的当前角速度
        :return:返回一个dt时间内的速度窗口
        """
        # 产生速度空间窗口
        # 最大速度，最大角速度限制
        vwlimit = [self.min_speed, self.max_speed, -self.max_yawrate, self.max_yawrate]
        # 基于当前速度、最大加速度、最大角加速度的速度限制
        vwmove = [v - self.max_accel * self.dt, v + self.max_accel * self.dt,
                  w - self.max_dyawrate * self.dt, w + self.max_dyawrate * self.dt]
        # 取交集
        vw = [max(vwlimit[0], vwmove[0]), min(vwlimit[1], vwmove[1]),
              max(vwlimit[2], vwmove[2]), min(vwlimit[3], vwmove[3])]
        # print("vw:{}".format(vw))
        return vw

    def prediction(self, x, y, v, w, alpha):
        """
        根据当前位置、朝向计算以某一v，w元组运动后，下一时间点的坐标、朝向
        :param x: 世界坐标系下当前横坐标
        :param y: 世界坐标系下当前纵坐标
        :param v: 当前速度
        :param w: 当前角速度
        :param alpha: 世界坐标系下当前机器人朝向
        :return: dt时间后下一目标横纵坐标
        """

        dt = self.dt
        # 当w极小时，简化计算
        if w != 0:
            if v / w >= 10000:
                w = 0

        if w == 0:
            prex = x + v * dt * math.cos(alpha)
            prey = y + v * dt * math.sin(alpha)
            prealpha = alpha

        else:
            theta = alpha
            if w >= 0:
                theta += math.pi / 2
            else:
                theta -= math.pi / 2

            # 计算瞬时转动半径
            r = (v / abs(w))
            # 根据无横移的轮式机器人模型预测下一时刻的坐标及朝向信息
            prex = x + r * math.cos(theta)
            prey = y + r * math.sin(theta)
            prex = prex + r * math.cos(theta + math.pi + w * dt)
            prey = prey + r * math.sin(theta + math.pi + w * dt)

            if w >= 0:
                prealpha = math.pi + theta + w * dt + math.pi / 2
            else:
                prealpha = math.pi + theta + w * dt - math.pi / 2

        return prex, prey, prealpha

    def gevaluate(self, prex, prey, prealpha, goal_x, goal_y):
        """
        计算预测位置、朝向与阶段目标点间的朝向损失
        :param prex: 预测下一时间点的横坐标
        :param prey: 预测下一时间点的纵坐标
        :param prealpha: 预测下一时间点的朝向角度
        :param goal_x: 当前阶段目标点横坐标
        :param goal_y: 当前阶段目标点纵坐标
        :return:  朝向损失
        """

        dx = goal_x - prex
        dy = goal_y - prey
        # 预测下一时刻所在位置与当前阶段目标点夹角的朝向损失
        delta = abs(math.atan2(dy, dx) - prealpha)
        # 损失函数归一化
        evaluation = abs(math.atan2(math.sin(delta), math.cos(delta))) / math.pi

        return evaluation

    def vevaluate(self, v):
        """
        计算速度损失，使速度尽可能大
        :param v: 当前评估的v,w元组中的速度v
        :return: 速度损失
        """
        # 归一化速度损失
        evaluation = (self.max_speed - v) / self.max_speed

        return evaluation

    def oevaluate(self, prex, prey, obs):
        """
        根据障碍点坐标信息计算障碍物损失
        :param prex: 预测下一时间点位置横坐标
        :param prey: 预测下一时间点位置纵坐标
        :param obs: 障碍物信息
        :return: 障碍物损失
        """

        obstree = KDTree(np.vstack((obs[0],obs[1])).T)

        # 利用kdtree快速计算预测点与距离它最近的障碍点的距离
        distance, index = obstree.query(np.array([prex, prey]))

        # 建立障碍物距离损失模型，接近障碍物时损失为无穷，较远时随距离增加快速下降
        evaluation = self.robot_size / (
                    distance - self.robot_size) if distance > self.robot_size else np.inf
        return evaluation

    def plan(self, x, goal, ob):
        """
        在速度窗口中采样，分别评估其总体损失并选取最优v,w组合
        :param x: 机器人当前位置及速度信息
        :param goal[0]: 阶段目标点横坐标
        :param goal[1]: 阶段目标点纵坐标
        :param ob: 障碍物信息
        :return: 最优控制参数元组v,w
        """
        # 部分参数赋初值
        min_eva = 1000000
        bestv = 0
        bestw = 0
        bestx = 0
        besty = 0
        besta = 0
        vw = self.window(x[3], x[4])

        start_time = time.time()
        # 对速度窗口、加速度窗口采样，评估以该[v,w]移动至下一时刻的总体损失
        for v in np.append(np.arange(vw[0], vw[1], (vw[1] - vw[0]) / self.vstep), vw[1]):
            for w in np.append(np.arange(vw[2], vw[3], (vw[3] - vw[2]) / self.wstep), [0, vw[3]]):
                prex, prey, prealpha = self.prediction(x[0], x[1], v, w, x[2])
                gevaluation = self.to_goal_cost_gain * self.gevaluate(prex, prey, prealpha, goal[0], goal[1])
                vevaluation = self.speed_cost_gain * self.vevaluate(v)
                oevaluation = self.obstacle_cost_gain * self.oevaluate(prex, prey, ob)
                # print(gevaluation, vevaluation, oevaluation)
                evaluation = gevaluation + vevaluation + oevaluation
                # 当总体损失小于记录的最低损失，更新bestv， bestw
                if evaluation <= min_eva:
                    min_eva = evaluation
                    bestv = v
                    bestw = w
                    bestx = prex
                    besty = prey
                    besta = prealpha
                    obs_score = oevaluation

        # print("The score in obs: ", obs_score, "!!!!!!!!!!!!!!!!")
        now_v = x[3]
        # print('bestv', bestv, 'nowv', now_v)

        # 若得到的bestv为负，说明此时向前移动会导致碰撞，故以较小速度后退
        # if bestv < 0.02:
        #    print('I am going to crash\n\n')
        #    return -0.05, 0, 0, 0, 0
        # if -0.5 < bestw < 0.5:
        #         bestw = 0
        # 若检测到即将到达一个阶段目标点且此处存在较大转弯角度，则以一定加速度减速，防止“冲过头”
        if goal[2] == 0.001 and math.hypot(goal[0] - x[0], goal[1] - x[1]) < self.threshold * 5 and now_v > self.max_speed / 3:
                bestv = min(now_v - self.max_accel / 5 * self.dt, bestv)
                print("\n\n\n\n\n\n\n\n\n")
                print('I am going to turn!!!!!')
                print("\n\n\n\n\n\n")

        if goal[2] == 0.001 and math.hypot(goal[0] - x[0], goal[1] - x[1]) < self.threshold * 2 and now_v > self.max_speed / 5:
                bestv = min(now_v - self.max_accel / 7 * self.dt, bestv)
                print('almost achieve turning')
                
        if goal[2] == 0.002 and math.hypot(goal[0] - x[0], goal[1] - x[1]) < self.threshold and now_v > self.max_speed / 5:
                bestv = min(now_v - self.max_accel / 7 * self.dt, bestv)
                bestw = 0
                print('almost achieve turning')
          
        u = [bestv, bestw]
        # print("Best Plan:", [bestv, bestw, bestx, besty, besta])

        end_time = time.time()
        # print("time:", end_time-start_time)
        return bestv, bestw, bestx, besty, besta
