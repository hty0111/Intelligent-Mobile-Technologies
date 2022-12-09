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
    def __init__(self, x=0.0, y=0.0, g=0.0, parent=-1):
        self.x = x
        self.y = y
        self.g = g
        self.h = np.inf
        self.f = 0.0
        self.parent = parent

    def get_HF(self, goal_x, goal_y):
        self.h = math.hypot(goal_x - self.x, goal_y - self.y)
        self.f = self.h + self.g


class Astar(object):
    def __init__(self, obstacle_x, obstacle_y, grid_size, robot_radius):
        self.avoid_dist = grid_size
        self.robot_radius = robot_radius
        self.current_node = 0
        self.close_set = []
        self.open_set = []
        self.obs_tree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        self.step_size = 0.5
        self.goal_threshold = 0.5
        self.MAKE_STRAIGHT = False

    def plan(self, start_x, start_y, goal_x, goal_y):
        start_node = Node(start_x, start_y, 0, 0)
        start_node.get_HF(goal_x, goal_y)
        # start_node.parent = start_node
        self.open_set.append(start_node)
        loop = 1
        while True:
            min_f = np.inf
            min_node = start_node
            for element in self.open_set:
                if element.f < min_f:
                    min_f = element.f
                    min_node = element
            self.open_set.remove(min_node)
            self.close_set.append(min_node)
            if min_node.h < self.goal_threshold:
                break
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    next_node = Node(min_node.x + i * self.step_size, min_node.y + j * self.step_size, min_node.g 
                                     + math.hypot(i * self.step_size, j * self.step_size))
                    next_node.get_HF(goal_x, goal_y)
                    distance, index = self.obs_tree.query(np.array([next_node.x, next_node.y]))
                    if distance <= self.avoid_dist + self.robot_radius / 2:
                        continue
                    if self.is_closeset(next_node):
                        continue
                    if not self.is_openset(next_node):
                        next_node.parent = min_node
                        self.open_set.append(next_node)
                    else:
                        if next_node.f < (self.is_openset(next_node)).f:
                            self.open_set.remove(self.is_openset(next_node))
                            next_node.parent = min_node
                            self.open_set.append(next_node)
            loop += 1
            if loop > 500:
                print("Cannot find path!")
                return [], []
    
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

        return path_x, path_y

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
        newpath_x, newpath_y = [path_x[0]], [path_y[0]]
        i = 0
        while True:
            j = 0
            for j in np.arange(len(path_x) - 1, i, -1):
                if not self.check_obs(path_x[i], path_y[i], path_x[j], path_y[j]):
                    newpath_x.append(path_x[j])
                    newpath_y.append(path_y[j])
                    i = j
                    break
            if j == len(path_x) - 1 and i == j - 1:
                break
            elif i == len(path_x) - 1:
                break

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

        step_size = self.avoid_dist
        steps = np.round(dist / step_size) - 1

        # print("dist: ", dist, "step: ", step_size)
        start_x += step_size * np.cos(angle)
        start_y += step_size * np.sin(angle)
        goal_x -= step_size * np.cos(angle)
        goal_y -= step_size * np.sin(angle)

        for i in range(int(steps)):
            distance, index = self.obs_tree.query(np.array([start_x, start_y]))
            if distance <= self.avoid_dist + self.robot_radius / 2:
                # print("dist_obs: ", distance)
                return True
            start_x += step_size * np.cos(angle)
            start_y += step_size * np.sin(angle)

        # check for goal point
        distance, index = self.obs_tree.query(np.array([goal_x, goal_y]))
        if distance <= self.avoid_dist:
            return True

        return False