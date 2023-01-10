#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @Description: A* planner
# @version: v1.0
# @Author: HTY
# @Date: 2022-07-05 22:58:37

from scipy.spatial import KDTree
import numpy as np
import math


class Node(object):
    def __init__(self, x, y, goal_x, goal_y, parent=-1):
        self.x = x
        self.y = y
        self.g = 0.0
        self.h = abs(goal_x - self.x) + abs(goal_y - self.y)
        self.f = self.h + self.g
        self.parent = parent

    def set_G(self, g):
        self.g = g
        self.f = self.h + self.g


class Astar(object):
    def __init__(self, obstacle_x, obstacle_y, grid_size, robot_radius):
        self.avoid_dist = 0.35
        self.robot_radius = robot_radius
        self.current_node = 0
        self.close_set = []
        self.open_set = []
        self.obs_tree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        self.step_size = 0.2
        self.goal_threshold = 0.3
        self.obstacle_x = obstacle_x
        self.obstacle_y = obstacle_y
        self.MAKE_STRAIGHT = True

    def plan(self, start_x, start_y, goal_x, goal_y):
        start_node = Node(start_x, start_y, goal_x, goal_y, 0)
        # start_node.parent = start_node
        self.open_set.append(start_node)
        loop = 1
        while True:
            # 遍历open_set，查找F最小的节点作为当前处理的节点
            min_f = np.inf
            min_node = start_node
            for element in self.open_set:
                if element.f < min_f:
                    min_f = element.f
                    min_node = element
            #         print(f"min node: {min_node}")
            # print(f"min_node_h: {min_node.h}, x: {min_node.x}, y: {min_node.y}")
            # print(f"open set: {self.open_set}")
            if not self.open_set:   # open_set为空，找不到路径
                break
            else:
                self.open_set.remove(min_node)
                self.close_set.append(min_node)
            if min_node.h < self.goal_threshold:    # 找到目标
                break

            # 遍历相邻的所有方格
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:   # 忽略当前方格
                        continue
                    next_node = Node(min_node.x + i * self.step_size, min_node.y + j * self.step_size, goal_x, goal_y)

                    # 判断障碍物
                    distance, index = self.obs_tree.query(np.array([next_node.x, next_node.y]))
                    if distance <= self.avoid_dist:
                        continue

                    if self.is_closeset(next_node):
                        continue
                    if not self.is_openset(next_node):
                        next_node.parent = min_node
                        next_node.set_G(min_node.g + math.hypot(i * self.step_size, j * self.step_size))
                        self.open_set.append(next_node)
                    else:   # in open_set
                        if min_node.g + math.hypot(i * self.step_size, j * self.step_size) < next_node.g:
                            next_node.parent = min_node
                            next_node.set_G(min_node.g + math.hypot(i * self.step_size, j * self.step_size))
            loop += 1
    
        # generate path
        path_x = [goal_x]
        path_y = [goal_y]
        reserve_node = self.close_set[-1]
        while reserve_node.parent != start_node:
            path_x.append(reserve_node.x)
            path_y.append(reserve_node.y)
            reserve_node = reserve_node.parent
        path_x.append(start_x)
        path_y.append(start_y)
        path_x.reverse()
        path_y.reverse()

        # make path straight
        if self.MAKE_STRAIGHT:
            path_x, path_y = self.make_straight(path_x, path_y)
            # path_x, path_y = self.optimize(path_x, path_y)

        turning = []
        for l in range(1,len(path_x)-1):
            dx1 = path_x[l] - path_x[l-1]
            dy1 = path_y[l] - path_y[l-1]
            dx2 = path_x[l+1] - path_x[l]
            dy2 = path_y[l+1] - path_y[l]
            theta1 = math.atan2(dy1, dx1)
            theta2 = math.atan2(dy2, dx2)
            delta = min(math.pi - theta1 + theta2, math.pi + theta1 - theta2)
            if delta < 5*math.pi/6:
                turning.append(0.001)
            else:
                turning.append(0)
        turning.append(0.002)

        defult_turning = []
        for i in range(len(path_x)):
            defult_turning.append(0)

        print("A* plan successfully!")

        return path_x, path_y, defult_turning

    def is_openset(self, node):
        for opennode in self.open_set:
            if opennode.x == node.x and opennode.y == node.y:
                return opennode
        return False

    def is_closeset(self, node):
        for closenode in self.close_set:
            if closenode.x == node.x and closenode.y == node.y:
                return True
        return False

    def make_straight(self, path_x, path_y):
        # print(f"len:{len(path_x)}")
        newpath_x, newpath_y = [path_x[0]], [path_y[0]]
        i = 0
        while True:
            j = 0
            for j in np.arange(len(path_x) - 1, i + 1, -1):
                if not self.check_obs(path_x[i], path_y[i], path_x[j], path_y[j]):
                    # print(f"i:{i}, j:{j}")
                    newpath_x.append(path_x[j])
                    newpath_y.append(path_y[j])
                    i = j
                    break
            if j == len(path_x) - 1 and i == j - 1:
                break
            elif i == len(path_x) - 1:
                break
            if i != j:  # 说明这个点没法straight
                i += 1
                newpath_x.append(path_x[i])
                newpath_y.append(path_y[i])
            # print(len(path_x) - 1)  # 28
            # print(f"i: {i}, j: {j}")    # 17, 18

        return newpath_x, newpath_y

    def check_obs(self, start_x, start_y, goal_x, goal_y):
        """
        Check collision
        :param ix, iy, nx, ny: start point and end point
        :return: (bool)
        """
        dx = goal_x - start_x
        dy = goal_y - start_y
        angle = np.arctan2(dy, dx)
        dist = np.sqrt(np.square(dx) + np.square(dy))

        step_size = self.step_size
        steps = np.round(dist / step_size) - 1

        # print("dist: ", dist, "step: ", step_size)
        start_x += step_size * np.cos(angle)
        start_y += step_size * np.sin(angle)
        goal_x -= step_size * np.cos(angle)
        goal_y -= step_size * np.sin(angle)

        for i in range(int(steps)):
            distance, index = self.obs_tree.query(np.array([start_x, start_y]))
            if distance <= self.avoid_dist:
                # print("dist_obs: ", distance)
                return True
            start_x += step_size * np.cos(angle)
            start_y += step_size * np.sin(angle)

        # check for goal point
        distance, index = self.obs_tree.query(np.array([goal_x, goal_y]))
        if distance <= self.avoid_dist:
            return True

        return False

