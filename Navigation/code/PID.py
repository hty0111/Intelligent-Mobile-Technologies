import numpy as np
from rrt_smart import RRT
from vision import Vision
from action import Action
from debug import Debugger
from prm import PRM
import time
import math
from zss_debug_pb2 import Debug_Msgs, Debug_Msg, Debug_Arc


class PIDcontrol(object):
    def __init__(self):
        self.w_kp = 10
        self.w_ki = 0.05
        self.w_kd = 0
        self.v_kp = 0.9
        self.v_ki = 0.015
        # self.v_kp = 5
        # self.v_ki = 0.1
        self.v_kd = 0

        self.w_max = 5
        self.v_max = 1500
        self.v_bias = 1000
        self.k = -4.0 / np.pi
        self.b = 2.0

        self.limit = 30     # interval for recheck obstacles
        self.step_size = 400    # for exception

    def control(self, vision, action, path_x, path_y, isDynamic=False):
        # debugger = Debugger()

        cnt = 0
        while len(path_x) >= 1:
            # for i in np.arange(num_path - 1):
            blue_robot_0 = vision.blue_robot[0]
            target_x = path_x[0]
            target_y = path_y[0]
            new_betw = 100000  # 距离最大值
            new_ori = 100000  # 角度最大值
            gap_arcsum = 0
            betw_sum = 0
            current_arc = blue_robot_0.orientation  # 初始的角度信息采样
            target_path, target_ori = self.find_angle_path(
                blue_robot_0.x, blue_robot_0.y, target_x, target_y)
            gap_arc = target_ori - current_arc
            while gap_arc > np.pi:  # 将角度限制在-pi到pi中
                gap_arc = gap_arc - 2 * np.pi
            while gap_arc < -np.pi:
                gap_arc = gap_arc + 2 * np.pi
            last_gap_arc = gap_arc
            last_betw = np.sqrt(
                np.square(blue_robot_0.x - target_x) +
                np.square(blue_robot_0.y - target_y))  # 与目标位置的距离差

            while True:
                current_arc = blue_robot_0.orientation  # 初始的角度信息采样
                target_path, target_ori = self.find_angle_path(
                    blue_robot_0.x, blue_robot_0.y, target_x, target_y)
                gap_arc = target_ori - current_arc
                # 将角度限制在-pi到pi中
                while gap_arc > np.pi:
                    gap_arc = gap_arc - 2 * np.pi
                while gap_arc < -np.pi:
                    gap_arc = gap_arc + 2 * np.pi
                gap_diff = gap_arc - last_gap_arc  # 角度误差的差分
                # 防止差分在角度为pi或-pi附近时算错
                while gap_diff > np.pi:
                    gap_diff = gap_diff - 2 * np.pi
                while gap_diff < -np.pi:
                    gap_diff = gap_diff + 2 * np.pi
                vw = self.w_PIDctrl(gap_arc, gap_arcsum, gap_diff)
                # 更新角度误差累积和上一次角度误差
                gap_arcsum = gap_arcsum + gap_arc
                if gap_arcsum > 40:
                    gap_arcsum = 40
                elif gap_arcsum < -40:
                    gap_arcsum = -40
                last_gap_arc = gap_arc

                betw = np.sqrt(
                    np.square(blue_robot_0.x - target_x) +
                    np.square(blue_robot_0.y - target_y))  # 与目标位置的距离差
                betw_diff = betw - last_betw
                vx = self.v_PIDctrl(betw, betw_sum, betw_diff)

                # 为了在角度差很大时转向不飞出去，速度乘以一个限幅
                restrict = self.k * gap_arc + self.b
                if restrict > 1:
                    restrict = 1
                if restrict < 0:
                    restrict = 0
                vx = vx * restrict

                betw_sum = betw_sum + betw
                if abs(betw_sum) >= 100000:
                    betw_sum = 100000
                last_betw = betw
                action.sendCommand(vx=vx, vw=vw)
                if betw < 100:  # 判断到达目标位置附近
                    action.sendCommand(vx=0, vw=0)
                    # 丢弃已走过的点
                    if len(path_x) > 1:
                        path_x = path_x[1:]
                        path_y = path_y[1:]
                        # 画出下一个要去的点
                        # package = Debug_Msgs()
                        # debugger.draw_point(package, x=path_x[0], y=path_y[0])
                        # debugger.send(package)
                        break
                    else:
                        # finish one way
                        return [None], [None]

                cnt += 1
                # print(cnt)
                if cnt == self.limit and isDynamic:
                    return path_x, path_y

                time.sleep(0.001)
                # break

    def find_angle_path(self, x0, y0, x1, y1):
        """
        Calculate angle and distance in the same coordinate system
        """
        path = np.sqrt(np.square(x1 - x0) + np.square(y1 - y0))
        x = x1 - x0
        y = y1 - y0
        if x > 0:
            angle = np.arctan(y / x)
        elif y > 0:
            angle = np.pi + np.arctan(y / x)
        else:
            angle = -np.pi + np.arctan(y / x)
        return path, angle

    def w_PIDctrl(self, vwerr, vwerr_sum, lastvwerr):
        result = self.w_kp * vwerr + self.w_ki * vwerr_sum + self.w_kd * lastvwerr
        if result > self.w_max:
            result = self.w_max
        if result < -self.w_max:
            result = -self.w_max
        return result

    def v_PIDctrl(self, vxerr, vxerr_sum, lastvxerr):
        result = self.v_kp * vxerr + self.v_ki * vxerr_sum + self.v_kd * lastvxerr + self.v_bias
        if result > self.v_max:
            result = self.v_max
        if result < 0:
            result = 0
        return result

    def turn_around(self, vision, action):
        """
        Turn around and step forward
        """
        current_ori = vision.my_robot.orientation
        current_x, current_y = vision.my_robot.x. vision.my_robot.y
        target_x = np.array([current_x - self.step_size * np.cos(current_ori)])
        target_y = np.array([current_y - self.step_size * np.cos(current_ori)])
        self.control(vision, action, target_x, target_y, interrupt=False)