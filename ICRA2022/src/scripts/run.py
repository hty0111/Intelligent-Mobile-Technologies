import time
import argparse
import subprocess
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import math
import rospy
import rospkg

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist

from gazebo_simulation import GazeboSimulation
from global_planner import GlobalPlanner

# from rrt2 import RRT as Planner
from rrt_connect import RRTCONNECT as Planner
# from a_star import Astar as Planner
import dwa2

INIT_POSITION = [-2, 3, 1.57]  # in world frame
GOAL_POSITION = [0, 10]  # relative to the initial position


def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def path_coord_to_gazebo_coord(x, y):
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5

    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift

    return gazebo_x, gazebo_y


def mapCallback(msg):
    map_data = np.array(msg.data).reshape((-1, msg.info.height)).transpose()
    ox, oy = np.nonzero(map_data > 50)
    obstacle_x = (ox * msg.info.resolution + msg.info.origin.position.x).tolist()
    obstacle_y = (oy * msg.info.resolution + msg.info.origin.position.y).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test BARN navigation challenge')
    parser.add_argument('--world_idx', type=int, default=0)

    # TODO: TEST MAP 50, 150, 200
    parser.add_argument('--gui', action="store_true")
    parser.add_argument('--out', type=str, default="out.txt")
    args = parser.parse_args()

    ##########################################################################################
    ## 0. Launch Gazebo Simulation
    ##########################################################################################

    os.environ["JACKAL_LASER"] = "1"
    os.environ["JACKAL_LASER_MODEL"] = "ust10"
    os.environ["JACKAL_LASER_OFFSET"] = "-0.065 0 0.01"

    world_name = "BARN/world_%d.world" % args.world_idx
    print(">>>>>>>>>>>>>>>>>> Loading Gazebo Simulation with %s <<<<<<<<<<<<<<<<<<" % world_name)
    rospack = rospkg.RosPack()
    base_path = rospack.get_path('jackal_helper')

    launch_file = join(base_path, 'launch', 'gazebo_launch.launch')
    world_name = join(base_path, "worlds", world_name)

    gazebo_process = subprocess.Popen([
        'roslaunch',
        launch_file,
        'world_name:=' + world_name,
        'gui:=' + ("true" if args.gui else "false")
    ])
    time.sleep(5)  # sleep to wait until the gazebo being created

    rospy.init_node('gym', anonymous=True)  # , log_level=rospy.FATAL)
    rospy.set_param('/use_sim_time', True)

    # GazeboSimulation provides useful interface to communicate with gazebo  
    gazebo_sim = GazeboSimulation(init_position=INIT_POSITION)

    init_coor = (INIT_POSITION[0], INIT_POSITION[1])
    goal_coor = (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1])

    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)
    collided = True

    # check whether the robot is reset, the collision is False
    while compute_distance(init_coor, curr_coor) > 0.1 or collided:
        gazebo_sim.reset()  # Reset to the initial position
        pos = gazebo_sim.get_model_state().pose.position
        curr_coor = (pos.x, pos.y)
        collided = gazebo_sim.get_hard_collision()
        time.sleep(1)

    ##########################################################################################
    ## 1. Launch your navigation stack
    ## (Customize this block to add your own navigation stack)
    ##########################################################################################

    # TODO: WRITE YOUR OWN NAVIGATION ALGORITHMS HERE
    # get laser data : data = gazebo_sim.get_laser_scan()
    # publish your final control through the topic /cmd_vel using : gazebo_sim.pub_cmd_vel([v, w])
    # if the global map is needed, read the map files, e.g. /jackal_helper/worlds/BARN/map_files/map_pgm_xxx.pgm

    # # DWA example
    # launch_file = join(base_path, '..', 'jackal_helper/launch/move_base_DWA.launch')
    # nav_stack_process = subprocess.Popen([
    #     'roslaunch',
    #     launch_file,
    # ])
    launch_file = join(base_path, '..', 'jackal_helper/launch/map.launch')
    map_process = subprocess.Popen([
        'roslaunch',
        launch_file
    ])

    global_planner = GlobalPlanner()
    # map_sub = rospy.Subscriber("/map", OccupancyGrid, mapCallback)
    rospy.sleep(2)

    path_planner = Planner(global_planner.obstacle_x, global_planner.obstacle_y, 1, 0.3)
    path_x, path_y, turning = path_planner.plan(INIT_POSITION[0], INIT_POSITION[1], GOAL_POSITION[0], GOAL_POSITION[1])
    plt.figure()
    plt.scatter(global_planner.obstacle_x, global_planner.obstacle_y, c="k")
    plt.scatter(path_x, path_y, c="r")
    plt.show()

    # Make sure your navigation stack receives a goal of (0, 10, 0), which is 10 meters away
    # along positive y-axis.
    import actionlib
    from geometry_msgs.msg import Quaternion
    from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction

    # nav_as = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    # mb_goal = MoveBaseGoal()
    # mb_goal.target_pose.header.frame_id = 'odom'
    # mb_goal.target_pose.pose.position.x = GOAL_POSITION[0]
    # mb_goal.target_pose.pose.position.y = GOAL_POSITION[1]
    # mb_goal.target_pose.pose.position.z = 0
    # mb_goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)
    #
    # nav_as.wait_for_server()
    # nav_as.send_goal(mb_goal)

    ##########################################################################################
    ## 2. Start navigation
    ##########################################################################################

    curr_time = rospy.get_time()
    pos = gazebo_sim.get_model_state().pose.position
    curr_coor = (pos.x, pos.y)
    print("init position: ", pos.x, pos.y)

    # check whether the robot started to move
    # while compute_distance(init_coor, curr_coor) < 0.1:
    #     curr_time = rospy.get_time()
    #     pos = gazebo_sim.get_model_state().pose.position
    #     curr_coor = (pos.x, pos.y)
    #     time.sleep(0.01)
    #     print("xxx")

    # start navigation, check position, time and collision
    start_time = curr_time
    start_time_cpu = time.time()
    collided = False

    v = 0
    w = 0

    get = 0
    config = dwa2.Config()
    vwplanner = dwa2.DWA(config)

    # my dwa!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # while compute_distance(goal_coor, curr_coor) > 1 and not collided and curr_time - start_time < 100:
    #     curr_time = rospy.get_time()
    #     pos = gazebo_sim.get_model_state().pose.position
    #     curr_coor = (pos.x, pos.y)
    #     # print("Time: %.2f (s), x: %.2f (m), y: %.2f (m)" % (curr_time - start_time, *curr_coor), end="\r")
    #     collided = gazebo_sim.get_hard_collision()

    for i in range(len(path_x)):

        goal_x, goal_y, goal_turn = path_x[i], path_y[i], turning[i]

        while True:
            # 获得当前世界坐标系中的机器人坐标、速度信息
            pos = gazebo_sim.get_model_state().pose.position
            ori = gazebo_sim.get_model_state().pose.orientation
            yaw = math.atan2(2 * (ori.w * ori.z + ori.x * ori.y), 1 - 2 * (ori.z * ori.z + ori.y * ori.y))

            start_x, start_y, start_yaw = pos.x, pos.y, yaw

            # 调用DWA算法规划最佳(v,w)组合
            # start_time = time.time()
            plan_state = [start_x, start_y, start_yaw, v, w]
            plan_goal = [goal_x, goal_y, goal_turn]
            plan_obs = [global_planner.obstacle_x, global_planner.obstacle_y]
            v, w, pre_x, pre_y, pre_yaw = vwplanner.plan(plan_state, plan_goal, plan_obs)
            gazebo_sim.pub_cmd_vel([v, w])
            # end_time = time.time()
            # print('time:', end_time-start_time)
            print("Move!\n")
            print("My position:", plan_state)
            print("My Goal", plan_goal)

            # 判断是否到达阶段目标点
            if math.sqrt((pos.x - goal_x) ** 2 + (pos.y - goal_y) ** 2) < 0.5:
                print("get to this goal")
                break


    ##########################################################################################
    ## 3. Report metrics and generate log
    ##########################################################################################

    print(">>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<")
    success = False
    if collided:
        status = "collided"
    elif curr_time - start_time >= 100:
        status = "timeout"
    else:
        status = "succeeded"
        success = True
    print("Navigation %s with time %.4f (s)" % (status, curr_time - start_time))

    path_file_name = join(base_path, "worlds/BARN/path_files", "path_%d.npy" % args.world_idx)
    path_array = np.load(path_file_name)
    path_array = [path_coord_to_gazebo_coord(*p) for p in path_array]
    path_array = np.insert(path_array, 0, (INIT_POSITION[0], INIT_POSITION[1]), axis=0)
    path_array = np.insert(path_array, len(path_array),
                           (INIT_POSITION[0] + GOAL_POSITION[0], INIT_POSITION[1] + GOAL_POSITION[1]), axis=0)
    path_length = 0
    for p1, p2 in zip(path_array[:-1], path_array[1:]):
        path_length += compute_distance(p1, p2)

    # Navigation metric: 1_success *  optimal_time / clip(actual_time, 4 * optimal_time, 8 * optimal_time)
    optimal_time = path_length / 2
    actual_time = curr_time - start_time
    nav_metric = int(success) * optimal_time / np.clip(actual_time, 4 * optimal_time, 8 * optimal_time)
    print("Navigation metric: %.4f" % (nav_metric))

    with open(args.out, "a") as f:
        f.write("%d %d %d %d %.4f %.4f\n" % (
        args.world_idx, success, collided, (curr_time - start_time) >= 100, curr_time - start_time, nav_metric))

    gazebo_process.terminate()
