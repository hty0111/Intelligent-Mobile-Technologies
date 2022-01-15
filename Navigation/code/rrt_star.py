from scipy.spatial import KDTree
import numpy as np
import random
import math
import time

from vision import Vision
from action import Action
from debug import Debugger

class RRT(object):
    def __init__(self, max_sample=500, step_size=200, prob_goal=0, search_radius=500):
        self.max_sample = max_sample
        self.step_size = step_size      # distance of extension
        self.prob_goal = prob_goal      # probability of choosing goal point when sampling
        self.search_radius = search_radius
        self.minx = -4500
        self.maxx = 4500
        self.miny = -3000
        self.maxy = 3000
        self.robot_size = 200
        self.avoid_dist = 200
        self.goal_threshold = 200
        
    def plan(self, vision, start_x, start_y, goal_x, goal_y, isSmooth=False):
        # Obstacles
        obstacle_x = []
        obstacle_y = []
        for robot_blue in vision.blue_robot:
            if robot_blue.visible and robot_blue.id > 0:
                obstacle_x.append(robot_blue.x)
                obstacle_y.append(robot_blue.y)
        for robot_yellow in vision.yellow_robot:
            if robot_yellow.visible:
                obstacle_x.append(robot_yellow.x)
                obstacle_y.append(robot_yellow.y)        
        # Obstacle KD Tree
        obs_tree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        # RRT tree
        rrt_x = [start_x]
        rrt_y = [start_y]
        # print(np.array([[start_x], [start_y]]).T)

        # RRT* algorithm
        road_map = [-1]
        final_points = []
        new_index = 1
        flag = 0
        path = []
        debugger = Debugger()

        while new_index < self.max_sample + 1:
            # Sample
            p = random.random()
            if p < self.prob_goal:
                sample_x, sample_y = goal_x, goal_y
                # print(p, "-------------------------")
            else:
                sample_x, sample_y = self.sampling()
            # print("sample: ", sample_x, sample_y, '\n')
            # Select the node in the RRT tree that is closest to the sample node
            nearest_x, nearest_y, nearest_index = self.nearestNode(sample_x, sample_y,  rrt_x, rrt_y)
            # print("nearest: ", nearest_x, nearest_y)
            # Create a new node accoding to the orientation
            new_x, new_y = self.extend(nearest_x, nearest_y, sample_x, sample_y, obs_tree)

            if new_x and new_y:
                road_map.append(nearest_index)
                rrt_x.append(new_x)
                rrt_y.append(new_y)
                road_map = self.rewire(rrt_x, rrt_y, new_x, new_y, road_map, obs_tree, new_index)
                # print(road_map)
                # print("new node: ", new_x, new_y)

                # if distance < threshold, close enough
                if math.sqrt((goal_x-new_x) ** 2 + (goal_y-new_y) ** 2) < self.goal_threshold:
                    final_points.append(new_index)
                    flag = 1        # find the goal
                    # choose final path from final points
                    path = self.final_path(rrt_x, rrt_y, road_map, final_points)
                # print(f'new insex {new_index}')
                new_index += 1
            
            # draw the tree and path
            debugger.my_draw_all(rrt_x, rrt_y, road_map, path)

        # print(final_points)
        path_x, path_y = [], []
        for i in path:
            path_x.append(rrt_x[i])
            path_y.append(rrt_y[i])

        if not isSmooth:
            path, path_x, path_y = self.make_straight(path, path_x, path_y, obs_tree)
        debugger.my_draw_all(rrt_x, rrt_y, road_map, path)
        
        if flag:
            print("Goal is found!")
            return path_x[1:], path_y[1:]
        else:
            print("Cannot find path!")

        return path
    
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
        return  nearest_x, nearest_y, nearest_index

    def extend(self, nearest_x, nearest_y, sample_x, sample_y, obs_tree):
        angle = math.atan2(sample_y - nearest_y, sample_x - nearest_x)
        new_x = nearest_x + math.cos(angle) * self.step_size
        new_y = nearest_y + math.sin(angle) * self.step_size

        if not self.check_obs(nearest_x, nearest_y, new_x, new_y, obs_tree):
            return new_x, new_y

        return None, None

    def check_obs(self, ix, iy, nx, ny, obs_tree):
        x = ix
        y = iy
        dx = nx - ix
        dy = ny - iy
        angle = math.atan2(dy, dx)
        dis = math.hypot(dx, dy)

        step_size = self.robot_size + self.avoid_dist
        steps = round(dis/step_size)
        for i in range(steps):
            distance, index = obs_tree.query(np.array([x, y]))
            if distance <= self.robot_size + self.avoid_dist:
                return True
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)

        # check for goal point
        distance, index = obs_tree.query(np.array([nx, ny]))
        if distance <= self.robot_size + self.avoid_dist:
            return True

        return False

    def rewire(self, rrt_x, rrt_y, new_x, new_y, road_map, obs_tree, new_index):
        # decide father node
        potential_father_index = []
        father_index = road_map[new_index]
        # calculate current cost
        current_cost = self.cal_cost(0, new_index, road_map, rrt_x, rrt_y)
        for i in range(new_index):
            # search for potential fahter node
            if self.cal_euler_dst(rrt_x[i], rrt_y[i], new_x, new_y) < self.search_radius:
                potential_father_index.append(i)
                # calculate new cost and decide new father node
                new_cost = self.cal_cost(0, i, road_map, rrt_x, rrt_y) + \
                           self.cal_euler_dst(rrt_x[i], rrt_y[i], new_x, new_y)
                if new_cost < current_cost and not self.check_obs(rrt_x[i], rrt_y[i], new_x, new_y, obs_tree):
                    road_map.pop()       # delete original path
                    father_index = i
                    road_map.append(father_index)   # new father node
                    current_cost = new_cost
        # rewire father node
        # print(potential_father_index, father_index)
        for i in potential_father_index:
            if  i != father_index:
                current_cost = self.cal_cost(0, i, road_map, rrt_x, rrt_y)
                new_cost = self.cal_cost(0, new_index, road_map, rrt_x, rrt_y) + \
                           self.cal_euler_dst(rrt_x[i], rrt_y[i], new_x, new_y)
                if new_cost < current_cost and not self.check_obs(rrt_x[i], rrt_y[i], new_x, new_y, obs_tree):
                    road_map[i] = new_index
        # print(road_map)
        return road_map

    def cal_euler_dst(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

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

    def make_straight(self, path, path_x, path_y, vision):
        newpath, newpath_x, newpath_y = [path[0]], [path_x[0]], [path_y[0]]
        i = 0
        while True:
            for j in np.arange(len(path) - 1, i, -1):
                if not self.check_obs(path_x[i], path_y[i], path_x[j], path_y[j], vision):
                    newpath.append(path[j])
                    newpath_x.append(path_x[j])
                    newpath_y.append(path_y[j])
                    i = j
                    break      
            if i == len(path) - 1:
                break

        return newpath, newpath_x, newpath_y

if __name__ == "__main__":
    vision = Vision()
    action = Action()
    # dynamic = ActionDynamic()
    debugger = Debugger()
    time.sleep(0.1)
    action.sendCommand(vx=0, vw=0)
    planner = RRT(max_sample=500, step_size=200, prob_goal=0.5)

    # 1. path planning
    start_x, start_y = vision.my_robot.x, vision.my_robot.y
    goal_x, goal_y = -4500, -3000
    # path_x[0], path_y[0] is the first target position
    path_x, path_y = planner.plan(vision, vision.my_robot.x, vision.my_robot.y, goal_x, goal_y)