from smooth_control import SmoothCtrlPredict
from rrt_star import RRT
from vision import Vision
from action import Action
from debug import Debugger
import time
import numpy as np
import matplotlib.pyplot as plt
from zss_debug_pb2 import Debug_Msgs, Debug_Msg, Debug_Arc


if __name__ == '__main__':
    vision = Vision()
    action = Action()
    # dynamic = ActionDynamic()
    debugger = Debugger()
    time.sleep(0.1)
    action.sendCommand(vx=0, vw=0)
    planner = RRT(max_sample=500, step_size=500, prob_goal=0.5)
    # trajectory = PIDcontrol()

    # 1. path planning
    start_x, start_y = vision.my_robot.x, vision.my_robot.y
    goal_x, goal_y = -3000, -2000
    # path_x[0], path_y[0] is the first target position
    path_x, path_y = planner.plan(vision, vision.my_robot.x, vision.my_robot.y, goal_x, goal_y, isSmooth=True)

    # 2. trajectory planning
    points_x = np.append(start_x, path_x)
    points_y = np.append(start_y, path_y)

    length = len(path_x)
    target_list = []
    for j in range(10):
        for i in range(length):
            target_list.append([path_x[i] / 1000, path_y[i] / 1000])
        path_x = np.flip(points_x)[1:]
        path_y = np.flip(points_y)[1:]
        points_x = np.flip(points_x)
        points_y = np.flip(points_y)
    # for i in range(len(path_x)):
    #     target_list.append([path_x[i] / 1000, path_y[i] / 1000])

    print(target_list)
    flag = 1
    # times = 10
    while True:
        trajectory = SmoothCtrlPredict((vision.my_robot.x / 1000, vision.my_robot.y / 1000, vision.my_robot.orientation), target_list)
        trajectory.predict()
        v, w = trajectory.feedback_vw()
        print(v, w)
        action.sendCommand(vx=v*1000, vw=w)
        if flag:
            x_list = [trajectory.trajectory_predict[i][0] for i in range(len(trajectory.trajectory_predict))]
            y_list = [trajectory.trajectory_predict[i][1] for i in range(len(trajectory.trajectory_predict))]
            flag = 0    

        if np.sqrt(np.square(vision.my_robot.x / 1000 - target_list[0][0]) + np.square(vision.my_robot.y / 1000 - target_list[0][1])) < 0.1:
            if len(target_list) == 1:            
                break
            else:
                target_list = target_list[1:]

        time.sleep(0.001)

    action.sendCommand(0, 0, 0)

    plt.figure("Smooth Control")
    plt.plot(x_list, y_list)
    plt.scatter([i / 1000 for i in path_x], [i / 1000 for i in path_y], c='r')
    # plt.show()
    plt.savefig("Smooth Control")

