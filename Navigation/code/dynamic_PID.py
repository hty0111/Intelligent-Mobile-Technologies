from PID import PIDcontrol
from rrt_smart import RRT
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
    planner = RRT(max_sample=1000, min_sample=0, step_size=500, prob_goal=0.8)
    trajectory = PIDcontrol()

    # 1. path planning
    start_x, start_y = vision.my_robot.x, vision.my_robot.y
    goal_x, goal_y = -3500, -2000
    # path_x[0], path_y[0] is the first target position
    path_x, path_y = planner.plan(vision, vision.my_robot.x, vision.my_robot.y, goal_x, goal_y)


    # 2. continuous trajectory planning and repeated path planning
    times = 10   # round trip for 5 times
    while times:
        action.controlObs(vision)
        try:
            path_x, path_y = trajectory.control(vision, action, path_x, path_y, isDynamic=True)
        except TypeError:
            print("UPD connection weak. Please try again.")
        else:
            if path_x[0] == None and path_y[0] == None:
                # 交换起点和终点
                temp = start_x; start_x = goal_x; goal_x = temp
                temp = start_y; start_y = goal_y; goal_y = temp
                path_x, path_y = planner.plan(vision, start_x, start_y, goal_x, goal_y)
                times -= 1
                continue
            if planner.check_dynamic_obs(vision, path_x, path_y):     # recheck & replan
                # path_x, path_y = planner.plan(vision, vision.my_robot.x, vision.my_robot.y, goal_x, goal_y)
                limit = 5   # max times of replan
                for i in np.arange(limit):
                    current_ori = vision.my_robot.orientation
                    path_x, path_y = planner.plan(vision, vision.my_robot.x, vision.my_robot.y, goal_x, goal_y)
                    _, plan_ori = trajectory.find_angle_path(vision.my_robot.x, vision.my_robot.y, path_x[0], path_y[0])

                    if np.abs(current_ori - plan_ori) < np.pi / 2:
                        break
                    action.sendCommand(vx=0, vw=0)

    action.sendCommand(vx=0, vw=0)

