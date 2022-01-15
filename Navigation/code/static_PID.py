from PID import PIDcontrol
from rrt_star import RRT
from vision import Vision
from action import Action
from debug import Debugger
from prm import PRM
import time
import math
import numpy as np
from zss_debug_pb2 import Debug_Msgs, Debug_Msg, Debug_Arc


if __name__ == '__main__':
    vision = Vision()
    action = Action()
    # dynamic = ActionDynamic()
    debugger = Debugger()
    time.sleep(0.1)
    action.sendCommand(vx=0, vw=0)
    planner = RRT(max_sample=500, step_size=200, prob_goal=0.5)
    trajectory = PIDcontrol()

    # 1. path planning
    start_x, start_y = vision.my_robot.x, vision.my_robot.y
    goal_x, goal_y = -3500, -2000
    # path_x[0], path_y[0] is the first target position
    path_x, path_y = planner.plan(vision, vision.my_robot.x, vision.my_robot.y, goal_x, goal_y)


    # 2. continuous trajectory planning and repeated path planning
    points_x = np.append(start_x, path_x)
    points_y = np.append(start_y, path_y)
    times = 10   # round trip for 5 times
    while times:
        trajectory.control(vision, action, points_x[1:], points_y[1:])
        # 交换起点和终点
        points_x = np.flip(points_x)
        points_y = np.flip(points_y)
        times -= 1

    action.sendCommand(vx=0, vw=0)

