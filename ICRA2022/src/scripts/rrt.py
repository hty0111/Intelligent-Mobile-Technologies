from scipy.spatial import KDTree
import numpy as np
import random
import math
import time


class Node(object):
    """
    定义节点，包括坐标、索引、子节点、父节点、转弯点、与其父节点之间距离等属性
    用以在线性表中存储树节点
    """
    def __init__(self, data, index):
        self.data = data
        self.children = []
        self.parent = None
        self.index = index
        self.turning = 0
        self.par_dis = 0

    # 添加子节点函数
    def add_chid(self,obj):
        self.children.append(obj)

    # 添加父节点函数
    def seek_parent(self,obj):
        self.parent = obj

    # 提取节点坐标函数
    def get_data(self):
        return self.data

    # 提取节点索引函数
    def get_index(self):
        return self.index

    def cal_par_dis(self):
        self.par_dis = math.hypot(self.data[0] - self.parent.data[0], self.data[1] - self.parent.data[1])

    def cal_dis(self):
        node = self
        len = 0
        while node != None:
            len += node.par_dis
            node = node.parent
        return len


class RRT(object):
    """
    初始化采样上限、生长步长、膨胀半径、地图范围、优化测试半径等
    """

    def __init__(self, plan_ox,plan_oy,plan_grid_size,plan_robot_radius):
        self.N_SAMPLE = 1000
        self.minx = -4.5
        self.maxx = 0
        self.miny = 0
        self.maxy = 10
        self.robot_size = plan_robot_radius
        self.avoid_dist = 0.05
        self.step = 0.3
        self.obstacle_x = plan_ox
        self.obstacle_y = plan_oy
        
        self.achieve_dis = 0.3
        self.rpt_dis = 0.2
        
        print("\n\n\nRRT object init successfully!!!\n\n")

    # rrt路径规划函数，输入起点和终点坐标，返回可行路径
    def plan(self, plan_sx,plan_sy,plan_gx,plan_gy):
        """

        :param plan_sx: 起点横坐标
        :param plan_sy: 起点纵坐标
        :param plan_gx: 终点横坐标
        :param plan_gy: 终点纵坐标
        :return: 规划路径节点横纵坐标，优化后路径节点横纵坐标，rrt树图及节点横纵坐标，优化后节点序列
        """
        # 记录开始时间
        start_time = time.time()
        # 初始化rrt树图，并将起点加入树图
        tree_map = []
        start = Node([plan_sx,plan_sy],0)
        tree_map.append(start)
        # # 导入障碍物坐标
        # obstacle_x = [-999999]
        # obstacle_y = [-999999]

        # 构建障碍物KDtree
        obstree = KDTree(np.vstack((self.obstacle_x, self.obstacle_y)).T)

        # 初始化节点xy坐标数组，并加入起点坐标
        rt_x, rt_y = [], []
        rt_x.append(plan_sx)
        rt_y.append(plan_sy)
        # 将起点加入rrt的KDtree
        rtree = KDTree(np.vstack((rt_x, rt_y)).T)
        # 初始化最终路径xy坐标数组
        path_x, path_y = [], []

        j = 0
        # 开始采样生长，判断是否找到路径或是否超过采样次数
        while self.check_end(rtree, plan_gx, plan_gy) and self.check_len(rt_x):
            # 采样并找到距离采样点最近的rrt树上的节点，从该节点出发向采样点生长一段距离step
            sample_x, sample_y = self.sampling(plan_gx, plan_gy, obstree)
            distance, index = rtree.query(np.array([sample_x, sample_y]))
            gx = rt_x[index]
            gy = rt_y[index]
            nx = gx + self.step*(sample_x-gx)/distance
            ny = gy + self.step*(sample_y-gy)/distance

            # 判断生长产生的新节点是否与障碍物有碰撞或是否与rrt树上现有节点重复（距离过近）
            if not self.check_obs(gx, gy, nx, ny, obstree, plan_sx, plan_sy) and not self.check_rpt(nx, ny, rtree):
                # print(9)
                # 将新节点加入rrt的KDtree
                j = j + 1
                rt_x.append(nx)
                rt_y.append(ny)
                rtree = KDTree(np.vstack((rt_x, rt_y)).T)
                # 将新节点加入rrt树图，并给新节点赋予父节点，给其父节点赋予子节点
                tree_map.append(Node([nx, ny], j))
                tree_map[j].seek_parent(tree_map[index])
                tree_map[index].add_chid(tree_map[j])
                tree_map[j].cal_par_dis()
        if not self.check_end(rtree, plan_gx, plan_gy):
                distance, index = rtree.query(np.array([plan_gx, plan_gy]))
                rt_x.append(plan_gx)
                rt_y.append(plan_gy)
                j += 1
                tree_map.append(Node([plan_gx, plan_gy],j))
                tree_map[j].seek_parent(tree_map[index])
                tree_map[index].add_chid(tree_map[j])
                tree_map[j].cal_par_dis()             
        # 初始化连通图
        road_map = []
        parent_map = []
        # 按照节点顺序将各个节点的子节点加入连通图
        for i in range(len(rtree.data)):
            road_map.append([child.get_index() for child in tree_map[i].children])
            parent_map.append(tree_map[i].parent.index if i !=0 else -1)
        # 将终点加入path
        path_x.append(tree_map[j].get_data()[0])
        path_y.append(tree_map[j].get_data()[1])
        # 根据父节点回溯path
        while i != 0:
            path_x.append(tree_map[i].parent.get_data()[0])
            path_y.append(tree_map[i].parent.get_data()[1])
            i = tree_map[i].parent.get_index()

        # 优化路径
        path_x = list(reversed(path_x))
        path_y = list(reversed(path_y))
        patho_x, patho_y = self.optimize(path_x, path_y, obstree)

        # 将优化后路径上的各节点加入optroad数组
        optroad = []
        for k in range(len(patho_x)):
            optroad.append(Node([patho_x[k], patho_y[k]], k))
        turning = []

        # 根据优化后路径上各节点与其前后节点连线的夹角值判断是否为一个转弯点
        for l in range(1,len(patho_x)-1):
            dx1 = patho_x[l] - patho_x[l-1]
            dy1 = patho_y[l] - patho_y[l-1]
            dx2 = patho_x[l+1] - patho_x[l]
            dy2 = patho_y[l+1] - patho_y[l]
            theta1 = math.atan2(dy1, dx1)
            theta2 = math.atan2(dy2, dx2)
            delta = min(math.pi - theta1 + theta2, math.pi + theta1 - theta2)
            if delta < 5*math.pi/6:
                optroad[l].turning = 1
                turning.append(0.001)
            else:
                turning.append(0)
        turning.append(0.002)

        # 记录结束时间
        end_time=time.time()
        # 对优化后的路径进行插值处理
        pathi_x, pathi_y = self.inter(patho_x, patho_y)

        defult_turning = []
        for i in range(len(pathi_x)):
            defult_turning.append(0)


        patho_x = list(reversed(patho_x))
        patho_y = list(reversed(patho_y))

        path_x = list(reversed(path_x))
        path_y = list(reversed(path_y))

        # pathi_x = list(reversed(pathi_x))
        # pathi_y = list(reversed(pathi_y))

        print("Plan Successfully!!!!!!!!!!!!!!!!!!!")

        # 返回优化、插值路径及转弯点判断（默认无）
        return pathi_x, pathi_y, defult_turning

    # 采样函数，输入目标点及障碍物的KDtree，返回一个合理的采样点的xy坐标，为了提高收敛速度，采样有50%概率进行全图随机采样，有50%概率直接将目标点作为本次采样点
    def sampling(self, plan_gx, plan_gy, obstree):
        """

        :param plan_gx: 终点横坐标
        :param plan_gy: 终点纵坐标
        :param obstree: 障碍物的KDtree
        :return: 采样点横纵坐标
        """
        sample_x = None
        sample_y = None
        cc = 1
        # cc用来保证每次采到且只采一个合理点
        while cc:
            # 50%概率全图随机采样
            if random.random() < 0.5:
                tx = (random.random() * (self.maxx - self.minx)) + self.minx
                ty = (random.random() * (self.maxy - self.miny)) + self.miny

                distance, index = obstree.query(np.array([tx, ty]))
                # 判断采样点是否在障碍物范围，防止rrt树生长方向朝向障碍物
                if distance >= self.robot_size + self.avoid_dist:
                    sample_x = tx
                    sample_y = ty
                    cc = 0

            # 50%概率将目标点设为本次采样点
            else:
                sample_x = plan_gx
                sample_y = plan_gy
                cc = 0

        return sample_x, sample_y

    # 检测是否找到可行路径函数，若找到则返回False，否则返回True
    def check_end(self, rtree, plan_gx, plan_gy):
        """

        :param rtree: rrt的KDtree
        :param plan_gx: 终点横坐标
        :param plan_gy: 终点纵坐标
        :return: True or False
        """
        distance, index = rtree.query(np.array([plan_gx, plan_gy]))
        # 判断与终点距离是否小于一定阈值
        if distance >= self.achieve_dis:
            return True

        return False

    # 检测两点之间连线是否与障碍物发生碰撞，若有碰撞则返回True，否则返回False
    def check_obs(self, ix, iy, nx, ny, obstree, start_x, start_y):
        """

        :param ix: 连线起点横坐标
        :param iy: 连线起点纵坐标
        :param nx: 连线终点横坐标
        :param ny: 连线终点纵坐标
        :param obstree: 障碍物的KDtree
        :return: True or False
        """
        x = ix
        y = iy
        dx = nx - ix
        dy = ny - iy
        angle = math.atan2(dy, dx)
        dis = math.hypot(dx, dy)
        # step_size = self.robot_size + self.avoid_dist
        step_size = 0.2
        steps = round(dis/step_size)
        # 两点间逐步检测
        for i in range(steps):
            distance, index = obstree.query(np.array([x, y]))
            if distance <= self.robot_size + self.avoid_dist and (x, y) != (start_x, start_y):
                return True
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)
        # 检测终点
        distance, index = obstree.query(np.array([nx, ny]))
        if distance <= self.robot_size + self.avoid_dist:
            return True

        return False

    # 判断新生节点是否与rrt树上已有节点重复（距离小于一定阈值），如果重复则返回True，否则返回False
    def check_rpt(self, nx, ny, rtree):
        """

        :param nx: 节点横坐标
        :param ny: 节点纵坐标
        :param rtree: rrt的KDtree
        :return: True or False
        """
        distance, index = rtree.query(np.array([nx, ny]))
        if distance <= self.rpt_dis:
            return True

        return False

    # 检查rrt树节点数是否超范围，如果超过预设上限则返回False，否则返回Ture
    def check_len(self, rt_x):
        """

        :param rt_x: rrt树节点的横坐标数组
        :return: True or False
        """
        if len(rt_x) >= self.N_SAMPLE:
            return False
        return True

    # 路径优化函数，避免不必要的小转弯减少折线数，输入待优化路径的节点xy坐标及障碍物KDtree，返回优化后路径的节点xy坐标
    def optimize(self, path_x, path_y, obstree):
        """

        :param path_x: 规划路径上节点的横坐标序列
        :param path_y: 规划路径上节点的纵坐标序列
        :param obstree: 障碍物的KDtree
        :return: 优化后路径上节点的横纵坐标序列
        """
        # 如果路径过短，无法优化
        if len(path_x) <= 3:
            return path_x, path_y

        i = 0
        flag = 0
        while True:
            # 寻找距离此次优化起点最远的且连线不与障碍物发生碰撞的节点
            for j in range(i+2, len(path_x)):
                if self.check_obs(path_x[i], path_y[i], path_x[j], path_y[j], obstree, path_x[0], path_y[0]):
                    flag = 1
                    break
            # 记录本次优化的起点索引
            sindex = i
            # 如果遍历整条路径上的点均无碰撞，则本次优化的终点即为原路径终点，否则本次优化终点索引为找到的第一个发生碰撞的节点的上一个节点的索引
            if j == len(path_x) - 1 and not flag:
                gindex = j
            else:
                gindex = j-1
            nsx = path_x[gindex]
            # 更新路径，抛弃本次优化起点终点之间的所有节点，直接将二者相连
            opath_x, opath_y = [], []
            opath_x.extend(path_x[:sindex])
            opath_y.extend(path_y[:sindex])
            opath_x.append(path_x[sindex])
            opath_y.append(path_y[sindex])
            opath_x.extend(path_x[gindex:])
            opath_y.extend(path_y[gindex:])
            path_x = opath_x
            path_y = opath_y

            i = path_x.index(nsx)
            flag = 0
            # 优化结束则跳出循环
            if i >= len(path_x)-2:
                break

        return path_x, path_y

    # 路径插值函数，对路径相邻两点之间进行逐步打点，输入待插值路径节点xy坐标，返回插值后路径节点xy坐标
    def inter(self, path_x, path_y):
        """

        :param path_x: 待插值的路径上节点的横坐标序列
        :param path_y: 待插值的路径上节点的纵坐标序列
        :return: 插值后的路径上节点的横纵坐标序列
        """
        # 初始化插值步长，新路径节点坐标序列
        lstep = 0.5
        npath_x, npath_y = [], []
        npath_x.append(path_x[0])
        npath_y.append(path_y[0])
        length = len(path_x)
        # 每相邻两点进行插值
        for i in range(length-1):
            sx = path_x[i]
            sy = path_y[i]
            gx = path_x[i+1]
            gy = path_y[i+1]
            dx = gx - sx
            dy = gy - sy
            dis = math.hypot(dx, dy)
            counts = math.floor(dis / lstep)

            # 逐步插值
            for j in range(1, counts):
                sx += lstep * dx / dis
                sy += lstep * dy / dis
                npath_x.append(sx)
                npath_y.append(sy)

            # 将插值点加入新路径序列
            npath_x.append(gx)
            npath_y.append(gy)

        return npath_x, npath_y



