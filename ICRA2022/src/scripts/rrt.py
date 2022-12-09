#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# @Description: RRT planner
# @version: v1.0
# @Author: HTY
# @Date: 2022-07-05 23:07:04

import rospy
from scipy.spatial import KDTree
import numpy as np
import random
import math


class RRT(object):
    min_x = -10
    max_x = 10
    min_y = -10
    max_y = 10

    def __init__(self, obstacle_x, obstacle_y, grid_size, robot_radius, max_point_num=500, max_sample_times=50000,
                 step_size=0.5, prob_goal=0.1, goal_threshold=1):
        self.obs_tree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)   # Obstacle KD Tree
        self.avoid_dist = grid_size
        self.robot_radius = robot_radius
        self.max_point_num = max_point_num
        self.max_sample_times = max_sample_times
        self.step_size = step_size  # distance of extension
        self.prob_goal = prob_goal  # probability of choosing goal point when sampling
        self.goal_threshold = goal_threshold
        self.MAKE_STRAIGHT = True

    def plan(self, start_x, start_y, goal_x, goal_y):
        # RRT tree
        rrt_x = [start_x]
        rrt_y = [start_y]
        # print(np.array([[start_x], [start_y]]).T)

        # RRT* algorithm
        road_map = [-1]  # father nodes
        final_points = []
        new_index = 1
        sample_times = 1
        find_flag = 0
        path = []

        while new_index < self.max_point_num + 1:
            if new_index % 50 == 0:
                rospy.loginfo("Index: %d | Loop: %d", new_index, sample_times)
            # Sample
            p = random.random()
            if p < self.prob_goal:
                sample_x, sample_y = goal_x, goal_y
                # print(p, "-------------------------")
            else:
                sample_x, sample_y = self.sampling()
            # print("sample: ", sample_x, sample_y)

            # Select the node in the RRT tree that is closest to the sample node
            nearest_x, nearest_y, nearest_index = self.nearestNode(sample_x, sample_y, rrt_x, rrt_y)
            # print("nearest: ", nearest_x, nearest_y)

            # Create a new node accoding to the orientation
            new_x, new_y = self.extend(nearest_x, nearest_y, sample_x, sample_y)
            # print("new: ", new_x, new_y)
            # print("----------------------------")

            if new_x and new_y:
                road_map.append(nearest_index)
                rrt_x.append(new_x)
                rrt_y.append(new_y)
                # print(road_map)
                # print("new node: ", new_x, new_y)

                # if distance < threshold, close enough
                if math.sqrt((goal_x - new_x) ** 2 + (goal_y - new_y) ** 2) < self.goal_threshold:
                    final_points.append(new_index)
                    find_flag = 1  # find the goal
                    # choose final path from final points
                    path = self.final_path(rrt_x, rrt_y, road_map, final_points)
                new_index += 1

            sample_times += 1
            if sample_times > self.max_sample_times:
                print("Reached max sampling times.")
                return [], []

            if find_flag:
                break

        # print(final_points)
        path_x, path_y = [], []
        for i in path:
            path_x.append(rrt_x[i])
            path_y.append(rrt_y[i])

        path_x.append(goal_x)
        path_y.append(goal_y)
        if self.MAKE_STRAIGHT:
            path_x, path_y = self.make_straight(path_x, path_y)
            path_x.append(goal_x)
            path_y.append(goal_y)

        if find_flag:
            print("Path is found!")
            return path_x, path_y
        else:
            print("Cannot find path!")

        return path_x, path_y

    def sampling(self):
        sample_x = (random.random() * (self.max_x - self.min_x)) + self.min_x
        sample_y = (random.random() * (self.max_y - self.min_y)) + self.min_y
        return sample_x, sample_y

    def nearestNode(self, sample_x, sample_y, rrt_x, rrt_y):
        """
        Find the nearest node on the tree to the sample point
        """
        min_dst = 999999
        nearest_x, nearest_y, nearest_index = 0, 0, 0
        for i in range(len(rrt_x)):
            dst = self.cal_euler_dst(rrt_x[i], rrt_y[i], sample_x, sample_y)
            math.sqrt((rrt_x[i] - sample_x) ** 2 + (rrt_y[i] - sample_y) ** 2)
            if dst < min_dst:
                min_dst = dst
                nearest_x = rrt_x[i]
                nearest_y = rrt_y[i]
                nearest_index = i
        return nearest_x, nearest_y, nearest_index

    def extend(self, nearest_x, nearest_y, sample_x, sample_y):
        """
        Extend given distance towards the sample point
        """
        angle = math.atan2(sample_y - nearest_y, sample_x - nearest_x)
        new_x = nearest_x + math.cos(angle) * self.step_size
        new_y = nearest_y + math.sin(angle) * self.step_size

        if not self.check_obs(nearest_x, nearest_y, new_x, new_y):
            return new_x, new_y

        return None, None

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
            if distance <= self.avoid_dist + self.robot_radius:
                # print("dist_obs: ", distance)
                return True
            start_x += step_size * np.cos(angle)
            start_y += step_size * np.sin(angle)

        # check for goal point
        distance, index = self.obs_tree.query(np.array([goal_x, goal_y]))
        if distance <= self.avoid_dist:
            return True

        return False

    def cal_euler_dst(self, x1, y1, x2, y2):
        """
        Calculate the Euler distance between two points
        :return: (float)
        """
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

    def cal_cost(self, start, end, road_map, rrt_x, rrt_y):
        """
        Calculate the Euler distance from start to end
        :param start: index of start node
        :param end: index of end node
        :param road_map: index of father node
        """
        father_index, current_index = road_map[end], end
        current_dst = 0
        # print("father index: ", father_index)
        # print("current index: ", current_index)
        # print("road_map: ", road_map)
        while father_index != road_map[start]:
            current_dst += self.cal_euler_dst(rrt_x[father_index], rrt_y[father_index],
                                              rrt_x[current_index], rrt_y[current_index])
            current_index = father_index
            father_index = road_map[father_index]
        return current_dst

    def final_path(self, rrt_x, rrt_y, road_map, final_points):
        path = []
        final_point = 0
        min_dst = 999999
        for i in final_points:
            current_dst = self.cal_cost(0, i, road_map, rrt_x, rrt_y)
            if current_dst < min_dst:
                min_dst = current_dst
                final_point = i
        path.append(final_point)
        father_index = road_map[final_point]
        while father_index != -1:
            path.append(father_index)
            father_index = road_map[father_index]
        path.reverse()
        return path

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


if __name__ == "__main__":
    ox = [1, 2, 3, 4, 5]
    oy = [1, 2, 3, 4, 5]

    planner = RRT(ox, oy, grid_size=0.3, robot_radius=0.8, max_point_num=100, max_sample_times=500, prob_goal=0.5)

    # 1. path planning
    start_x, start_y = 0, 0
    goal_x, goal_y = 8, 8
    # path_x[0], path_y[0] is the first target position
    path_x, path_y = planner.plan(start_x, start_y, goal_x, goal_y)
    print(path_x, path_y)
