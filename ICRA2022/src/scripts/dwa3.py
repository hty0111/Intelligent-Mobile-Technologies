from scipy.spatial import KDTree
import numpy as np
import math

pi = math.pi


class DWA(object):
    def __init__(self):
        self.vmax = 3000
        self.vmin = -3000
        self.wmax = 5
        self.wmin = -5
        self.avmax = 3000
        self.awmax = 5
        self.dt = 0.1
        self.gfactor = 2
        self.vfactor = 1.5
        self.ofactor = 8
        self.robot_size = 60
        self.vstep = 5
        self.wstep = 10




    def window(self, v, w):
        """
        根据当前v。w计算速度窗口
        :param v: 机器人坐标系下的当前速度
        :param w: 机器人坐标系下的当前角速度
        :return:返回一个dt时间内的速度窗口
        """
        # 产生速度空间窗口

        # 最大速度，最大角速度限制
        vwlimit = [self.vmin, self.vmax, self.wmin, self.wmax]
        # 基于当前速度、最大加速度、最大角加速度的速度限制
        vwmove = [v - self.avmax * self.dt, v + self.avmax * self.dt,
                  w - self.awmax * self.dt, w + self.awmax * self.dt]
        # 取交集
        vw = [max(vwlimit[0], vwmove[0]), min(vwlimit[1], vwmove[1]),
              max(vwlimit[2], vwmove[2]), min(vwlimit[3], vwmove[3])]
        print("vw:{}".format(vw))
        return vw


    def prediction(self, x, y, v, w, alpha):
        """
        根据当前位置、朝向计算以某一v，w元组运动后，下一时间点的坐标、朝向
        :param x: 世界坐标系下当前横坐标
        :param y: 世界坐标系下当前纵坐标
        :param v: 当前速度
        :param w: 当前角速度
        :param alpha: 世界坐标系下当前机器人朝向
        :return: dt时间后下一目标横纵坐标
        """

        dt = self.dt
        # 当w极小时，简化计算
        if w != 0:
            if v / w >= 10000:
                w = 0

        if w == 0:
            prex = x + v * dt * math.cos(alpha)
            prey = y + v * dt * math.sin(alpha)
            prealpha = alpha

        else:
            theta = alpha
            if w >= 0:
                theta += pi / 2
            else:
                theta -= pi / 2

            # 计算瞬时转动半径
            r = (v / abs(w))
            # 根据无横移的轮式机器人模型预测下一时刻的坐标及朝向信息
            prex = x + r * math.cos(theta)
            prey = y + r * math.sin(theta)
            prex = prex + r * math.cos(theta + pi + w * dt)
            prey = prey + r * math.sin(theta + pi + w * dt)

            if w >= 0:
                prealpha = pi + theta + w * dt + pi / 2
            else:
                prealpha = pi + theta + w * dt - pi / 2


        return prex, prey, prealpha

    def gevaluate(self, prex, prey, prealpha, goal_x, goal_y):
        """
        计算预测位置、朝向与阶段目标点间的朝向损失
        :param prex: 预测下一时间点的横坐标
        :param prey: 预测下一时间点的纵坐标
        :param prealpha: 预测下一时间点的朝向角度
        :param goal_x: 当前阶段目标点横坐标
        :param goal_y: 当前阶段目标点纵坐标
        :return:  朝向损失
        """

        dx = goal_x - prex
        dy = goal_y - prey
        # 预测下一时刻所在位置与当前阶段目标点夹角的朝向损失
        delta = abs(math.atan2(dy, dx) - prealpha)
        # 损失函数归一化
        evaluation = abs(math.atan2(math.sin(delta), math.cos(delta))) / math.pi

        return evaluation

    def vevaluate(self, v):
        """
        计算速度损失，使速度尽可能大
        :param v: 当前评估的v,w元组中的速度v
        :return: 速度损失
        """
        # 归一化速度损失
        evaluation = (self.vmax - v) / self.vmax

        return evaluation

    def oevaluate(self, prex, prey, vision):
        """
        根据障碍点坐标信息计算障碍物损失
        :param prex: 预测下一时间点位置横坐标
        :param prey: 预测下一时间点位置纵坐标
        :param vision: 包含client各物体位置信息
        :return: 障碍物损失
        """

        obstacle_x = [-999999]
        obstacle_y = [-999999]

        # 将当前障碍物所在坐标存入kd树
        for robot_blue in vision.blue_robot:
            if robot_blue.visible and robot_blue.id > 0:
                obstacle_x.append(robot_blue.x)
                obstacle_y.append(robot_blue.y)
        for robot_yellow in vision.yellow_robot:
            if robot_yellow.visible:
                obstacle_x.append(robot_yellow.x)
                obstacle_y.append(robot_yellow.y)
        obstree = KDTree(np.vstack((obstacle_x,obstacle_y)).T)

        # 利用kdtree快速计算预测点与距离它最近的障碍点的距离
        distance, index = obstree.query(np.array([prex,prey]))

        # 建立障碍物距离损失模型，接近障碍物时损失为无穷，较远时随距离增加快速下降
        evaluation = self.robot_size / (distance-self.robot_size) if distance>self.robot_size else np.inf
        return evaluation



    def vwplan(self, x, y, alpha, goal_x, goal_y, vision, vw, fgoal_x, fgoal_y, optroad, i):
        """
        在速度窗口中采样，分别评估其总体损失并选取最优v,w组合
        :param x: 机器人在世界坐标系中的当前横坐标
        :param y: 机器人在世界坐标系中的当前纵坐标
        :param alpha: 机器人在世界坐标系中的当前朝向角度
        :param goal_x: 阶段目标点横坐标
        :param goal_y: 阶段目标点纵坐标
        :param vision: 包含client各物体位置信息
        :param vw: 计算得到的当前速度窗口
        :param fgoal_x: 最终目标点横坐标
        :param fgoal_y: 最终目标点纵坐标
        :param optroad: 优化后规划路径的节点Node序列
        :param i: 阶段目标点处在optroad中的下标
        :return: 最优控制参数元组v,w
        """
        # 部分参数赋初值
        min_eva = 1000000
        bestv = 0
        bestw = 0

        # 对速度窗口、加速度窗口采样，评估以该[v,w]移动至下一时刻的总体损失
        for v in np.arange(vw[0], vw[1], (vw[1]-vw[0])/self.vstep):
            for w in np.append(np.arange(vw[2], vw[3], (vw[3]-vw[2])/self.wstep),0):
                prex, prey, prealpha = self.prediction(x,y,v,w,alpha)
                gevaluation = self.gfactor * self.gevaluate(prex, prey, prealpha, goal_x, goal_y)
                vevaluation = self.vfactor * self.vevaluate(v)
                oevaluation = self.ofactor * self.oevaluate(prex, prey, vision)
                # print(gevaluation, vevaluation, oevaluation)
                evaluation = gevaluation + vevaluation + oevaluation
                # 当总体损失小于记录的最低损失，更新bestv， bestw
                if evaluation <= min_eva:
                    min_eva = evaluation
                    bestv = v
                    bestw = w
        now_v = math.hypot(vision.my_robot.vel_x, vision.my_robot.vel_y)
        print('bestv', bestv, 'nowv', now_v)

        # 若得到的bestv为负，说明此时向前移动会导致碰撞，故以较小速度后退
        if bestv < 50:

            print('I am going to crash\n\n')
            return -100, 0

        # 若检测到即将到达一个阶段目标点且此处存在较大转弯角度，则以一定加速度减速，防止“冲过头”
        if optroad[i].turning == 1 and math.hypot(goal_x-vision.my_robot.x, goal_y-vision.my_robot.y)<1300 and now_v > 800:
            bestv=min(now_v-2500*self.dt,bestv)
            print('I am going to turn!!!!!')
        if optroad[i].turning == 1 and math.hypot(goal_x-vision.my_robot.x, goal_y-vision.my_robot.y)<700 and now_v > 300:
            bestv=min(now_v-1000*self.dt,bestv)
        if abs(bestw) > 1.5 and bestv > 600:
            bestv = min(now_v - 2500 * self.dt, bestv)
            bestw = max(abs(bestw), vw[3]*bestw/abs(bestw), vw[2]*bestw/abs(bestw))*bestw/abs(bestw)

        # 若检测到即将到达最终目标点，则以一定加速度减速
        if math.sqrt((fgoal_x-vision.my_robot.x)**2+(fgoal_y-vision.my_robot.y)**2)<1500 and now_v >1100:
          bestv=min(now_v-1000*self.dt,bestv)
          print('I am going to get to the goal!!!')
        if math.sqrt((fgoal_x-vision.my_robot.x)**2+(fgoal_y-vision.my_robot.y)**2)<500 and now_v >300:
          bestv=min(now_v-500*self.dt,bestv)
          print('I am really going to get to the goal!!!')

        return bestv, bestw