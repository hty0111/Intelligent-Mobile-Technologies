from scipy.spatial import KDTree
import numpy as np
import random
import math
import time

from vision import Vision
from action import Action
from debug import Debugger

class RRT(object):
    def __init__(self,
                 max_sample = 500,
                 min_sample = 0,
                 step_size = 500,
                 prob_goal = 0,
                 search_radius = 500):

        self.max_sample = max_sample
        self.min_sample = min_sample
        self.step_size = step_size  # distance of extension
        self.prob_goal = prob_goal  # probability of choosing goal point when sampling
        self.search_radius = search_radius
        self.minx = -4500
        self.maxx = 4500
        self.miny = -3000
        self.maxy = 3000
        self.robot_size = 200
        self.avoid_dist = 200
        self.goal_threshold = 500

    def plan(self, vision, start_x, start_y, goal_x, goal_y):
        # RRT tree
        rrt_x = np.array([start_x], dtype=int)
        rrt_y = np.array([start_y], dtype=int)
        # print(np.array([[start_x], [start_y]]).T)

        # RRT* algorithm
        road_map = np.array([-1], dtype=int)    # father nodes
        final_points = np.array([], dtype=int)
        new_index = 1
        flag = 0
        path = np.array([], dtype=int)
        debugger = Debugger()

        while new_index < self.max_sample + 1:
            # print(f'new index {new_index}')
            # Sample
            p = random.random()
            if p < self.prob_goal:
                sample_x, sample_y = goal_x, goal_y
            else:
                sample_x, sample_y = self.sampling()
            # print("sample: ", sample_x, sample_y, '\n')
            # Select the node in the RRT tree that is closest to the sample node
            nearest_x, nearest_y, nearest_index = self.nearestNode(
                sample_x, sample_y, rrt_x, rrt_y)
            # print("nearest: ", nearest_x, nearest_y)
            # Create a new node accoding to the orientation
            new_x, new_y = self.extend(nearest_x, nearest_y, sample_x,
                                       sample_y, vision)

            if new_x and new_y:
                road_map = np.append(road_map, nearest_index)
                rrt_x = np.append(rrt_x, new_x)
                rrt_y = np.append(rrt_y, new_y)
                road_map = self.rewire(rrt_x, rrt_y, new_x, new_y, road_map,
                                       vision, new_index)
                # print(road_map)
                # print("new node: ", new_x, new_y)

                # if distance < threshold, close enough
                if np.sqrt(np.square(goal_x - new_x) + np.square(goal_y - new_y)) < self.goal_threshold:
                    final_points = np.append(final_points, new_index)
                    flag = 1  # find the goal
                    # choose final path from final points
                    path, path_x, path_y = self.final_path(
                        rrt_x, rrt_y, road_map, final_points)
                    # make path straight
                    path, path_x, path_y = self.make_straight(
                        path, path_x, path_y, vision)
                    debugger.my_draw_all(rrt_x, rrt_y, road_map, path)

                    if new_index > self.min_sample:
                        if len(path_x) > 1:
                            print('find the goal!')
                            return path_x[1:], path_y[1:]
                        else:
                            return self.plan(vision, vision.my_robot.x, 
                                        vision.my_robot.y, goal_x, goal_y)
                    
                
                # print(new_index)
                new_index += 1
                
        print("Cannot find path!")
        return [], []

    def sampling(self):
        sample_x = (random.random() * (self.maxx - self.minx)) + self.minx
        sample_y = (random.random() * (self.maxy - self.miny)) + self.miny

        # RRT采样点不需要碰撞检测
        # distance, _ = obs_tree.query(np.array([sample_x, sample_y]))

        # if distance >= self.robot_size + self.avoid_dist:
        #     return sample_x, sample_y
        # else:       # With collision, sampling again
        #     return self.sampling(obs_tree)

        return sample_x, sample_y

    def nearestNode(self, sample_x, sample_y, rrt_x, rrt_y):
        """
        Find s nearest node on the tree to the sample point
        """
        min_dst = 999999
        nearest_x, nearest_y, nearest_index = 0, 0, 0
        for i in np.arange(len(rrt_x)):
            dst = self.cal_euler_dst(rrt_x[i], rrt_y[i], sample_x, sample_y)
            np.sqrt(np.square(rrt_x[i] - sample_x) + np.square(rrt_y[i] - sample_y))
            if dst < min_dst:
                min_dst = dst
                nearest_x = rrt_x[i]
                nearest_y = rrt_y[i]
                nearest_index = i
        return nearest_x, nearest_y, nearest_index

    def extend(self, nearest_x, nearest_y, sample_x, sample_y, vision):
        """
        Extend given distance towards the sample point
        """
        angle = np.arctan2(sample_y - nearest_y, sample_x - nearest_x)
        new_x = nearest_x + np.cos(angle) * self.step_size
        new_y = nearest_y + np.sin(angle) * self.step_size
        # print(f'new: {new_x, new_y}')

        if not self.check_obs(nearest_x, nearest_y, new_x, new_y, vision):
            return new_x, new_y

        return None, None

    def check_obs(self, ix, iy, nx, ny, vision):
        """
        Check collision
        :param ix, iy, nx, ny: two points
        :param obs_tree: info of obstacles
        :return: (bool)
        """
        # Obstacle KD Tree
        obstacle_x, obstacle_y = self.find_obs(vision)
        obs_tree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)

        x = ix
        y = iy
        dx = nx - ix
        dy = ny - iy
        angle = np.arctan2(dy, dx)
        # dis = math.hypot(dx, dy)
        dis = np.sqrt(np.square(x) + np.square(y))

        step_size = (self.robot_size + self.avoid_dist) / 2
        steps = np.round(dis / step_size)
        x += step_size * np.cos(angle) * 2
        y += step_size * np.sin(angle) * 2
        for i in np.arange(steps):
            distance, index = obs_tree.query(np.array([x, y]))
            if distance <= (self.robot_size + self.avoid_dist):
                return True
            x += step_size * np.cos(angle)
            y += step_size * np.sin(angle)

        # check for goal point
        distance, index = obs_tree.query(np.array([nx, ny]))
        if distance <= self.robot_size + self.avoid_dist:
            return True

        return False

    def rewire(self, rrt_x, rrt_y, new_x, new_y, road_map, vision,
               new_index):
        """
        Rewire the path for RRT*
        :param rrt_x, rrt_y: exist points on the map
        :param road_map: index of father node
        :param obs_tree: info of obstacles
        :return rosd_map: (ndarray) nodes of path
        """
        # decide father node
        potential_father_index = np.array([], dtype=int)
        father_index = road_map[new_index]
        # calculate current cost
        current_cost = self.cal_cost(0, new_index, road_map, rrt_x, rrt_y)
        for i in range(new_index):
            # search for potential fahter node
            if self.cal_euler_dst(rrt_x[i], rrt_y[i], new_x,
                                  new_y) < self.search_radius:
                potential_father_index = np.append(potential_father_index, i)
                # calculate new cost and decide new father node
                new_cost = self.cal_cost(0, i, road_map, rrt_x, rrt_y) + \
                           self.cal_euler_dst(rrt_x[i], rrt_y[i], new_x, new_y)
                if new_cost < current_cost and not self.check_obs(
                        rrt_x[i], rrt_y[i], new_x, new_y, vision):
                    road_map = road_map[: -1]       # delete original path
                    father_index = i
                    road_map = np.append(road_map, father_index)    # new father node
                    current_cost = new_cost
        # rewire father node
        # print(potential_father_index, father_index)
        for i in potential_father_index:
            if i != father_index:
                current_cost = self.cal_cost(0, i, road_map, rrt_x, rrt_y)
                new_cost = self.cal_cost(0, new_index, road_map, rrt_x, rrt_y) + \
                           self.cal_euler_dst(rrt_x[i], rrt_y[i], new_x, new_y)
                if new_cost < current_cost and not self.check_obs(
                        rrt_x[i], rrt_y[i], new_x, new_y, vision):
                    road_map[i] = new_index
        # print(road_map)
        return road_map

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
        father_index = road_map[end]
        current_index = end
        current_dst = 0

        while father_index != road_map[start]:
            current_dst += self.cal_euler_dst(rrt_x[father_index],
                                              rrt_y[father_index],
                                              rrt_x[current_index],
                                              rrt_y[current_index])
            current_index = father_index
            father_index = road_map[father_index]
        return current_dst

    def final_path(self, rrt_x, rrt_y, road_map, final_points):
        """
        Find a shortest path from all possible roads
        """
        path, path_x, path_y = np.array([[], [], []], dtype=int)
        final_point = 0
        min_dst = 999999

        # find a shortest path
        for i in final_points:
            current_dst = self.cal_cost(0, i, road_map, rrt_x, rrt_y)
            if current_dst < min_dst:
                min_dst = current_dst
                final_point = i
        path = np.append(path, final_point)
        path_x = np.append(path_x, rrt_x[final_point])
        path_y = np.append(path_y, rrt_y[final_point])
        father_index = road_map[final_point]
        while father_index != -1:
            path = np.append(path, father_index)
            path_x = np.append(path_x, rrt_x[father_index])
            path_y = np.append(path_y, rrt_y[father_index])
            father_index = road_map[father_index]
        path = np.flip(path)
        path_x = np.flip(path_x)
        path_y = np.flip(path_y)
        return path, path_x, path_y

    def make_straight(self, path, path_x, path_y, vision):
        newpath, newpath_x, newpath_y = np.array([[path[0]], [path_x[0]], [path_y[0]]], dtype=int)
        i = 0
        while True:
            for j in np.arange(len(path) - 1, i, -1):
                if not self.check_obs(path_x[i], path_y[i], path_x[j], path_y[j], vision):
                    newpath = np.append(newpath, path[j])
                    newpath_x = np.append(newpath_x, path_x[j])
                    newpath_y = np.append(newpath_y, path_y[j])
                    i = j
                    break      
            if i == len(path) - 1:
                break

        return newpath, newpath_x, newpath_y

    def find_obs(self, vision):
        """
        :return: position of obstacles
        """
        obstacle_x = np.array([], dtype=int)
        obstacle_y = np.array([], dtype=int)
        for robot_blue in vision.blue_robot:
            if robot_blue.visible and robot_blue.id > 0:
                obstacle_x = np.append(obstacle_x, robot_blue.x)
                obstacle_y = np.append(obstacle_y, robot_blue.y)
        for robot_yellow in vision.yellow_robot:
            if robot_yellow.visible:
                obstacle_x = np.append(obstacle_x, robot_yellow.x)
                obstacle_y = np.append(obstacle_y, robot_yellow.y)
        
        return obstacle_x, obstacle_y

    def check_dynamic_obs(self, vision, path_x, path_y):     
        check_points = np.array([vision.my_robot.x, vision.my_robot.y])
        for i in range(len(path_x)):
            check_points = np.vstack((check_points, [path_x[i], path_y[i]]))
        for i in range(len(check_points) - 1):
            # print(point)
            if self.check_obs(check_points[i][0], check_points[i][1], 
                                check_points[i+1][0], check_points[i+1][1], vision):
                # print(check_points[i][0], check_points[i][1], check_points[i+1][0], check_points[i+1][1])
                return True

        return False


if __name__ == "__main__":
    vision = Vision()
    rrt = RRT(max_sample=200)
    path = rrt.plan(vision=vision,
                    start_x=vision.my_robot.x,
                    start_y=vision.my_robot.y,
                    goal_x=-2400,
                    goal_y=1500)
    print(path)
