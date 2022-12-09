from scipy.spatial import KDTree
import numpy as np
import random
import math
import time


class Node1(object):
    """
    定义节点，包括坐标、索引、子节点、父节点、转弯点、与其父节点之间距离等属性
    由于是两棵rrt树，这里设置两组节点以便区分
    用以在线性表中存储树节点
    """
    def __init__(self,data,index):
        self.data = data
        self.children = []
        self.parent = None
        self.index = index
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

class Node2(object):
    """
    第二组节点
    """
    def __init__(self,data,index):
        self.data = data
        self.children = []
        self.parent = None
        self.index = index
        self.par_dis = 0

    def add_chid(self,obj):
        self.children.append(obj)

    def seek_parent(self,obj):
        self.parent = obj

    def get_data(self):
        return self.data

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

# 该组节点是为了判断是否为转弯点，主要利用turning这一属性
class Node(object):
    def __init__(self,data,index):
        self.data = data
        self.children = []
        self.parent = None
        self.index = index
        self.turning = 0

class RRTCONNECT(object):
    # 初始化采样上限、生长步长、膨胀半径、地图范围等
    def __init__(self, plan_ox, plan_oy, plan_grid_size, plan_robot_radius):
        self.N_SAMPLE = 1000
        self.minx = -4.5
        self.maxx = 0
        self.miny = 0
        self.maxy = 10
        self.robot_size = plan_robot_radius
        self.avoid_dist = 0
        self.step = 0.3
        self.obstacle_x = plan_ox
        self.obstacle_y = plan_oy

        self.achieve_dis = 0.3
        self.rpt_dis = 0.18

        print("\n\n\nRRT object init successfully!!!\n\n")

    # 双向rrt路径规划函数，输入起点和终点坐标，返回可行路径
    def plan(self, plan_sx, plan_sy, plan_gx, plan_gy):
        """

        :param plan_sx: 起点横坐标
        :param plan_sy: 起点纵坐标
        :param plan_gx: 终点横坐标
        :param plan_gy: 终点纵坐标
        :return: 规划路径节点横纵坐标，优化后路径节点横纵坐标，双向rrt树图及节点横纵坐标，优化后节点序列
        """
        # 记录开始时间
        start_time = time.time()
        # 初始化两个rrt树图，并将起点和终点分别加入两个树图
        tree1_map = []
        tree2_map = []
        start = Node1([plan_sx, plan_sy], 0)
        goal = Node2([plan_gx, plan_gy], 0)
        tree1_map.append(start)
        tree2_map.append(goal)
        # 导入障碍物坐标

        # 构建障碍物KDtree
        obstree = KDTree(np.vstack((self.obstacle_x, self.obstacle_y)).T)
        # 初始化两组节点xy坐标数组，并分别加入起点和终点坐标
        rt_x, rt_y = [], []
        rt_x1, rt_y1 = [], []
        rt_x1.append(plan_sx)
        rt_y1.append(plan_sy)
        rt_x2, rt_y2 = [], []
        rt_x2.append(plan_gx)
        rt_y2.append(plan_gy)
        # 将起点和终点分别加入两个rrt的KDtree
        rtree1 = KDTree(np.vstack((rt_x1, rt_y1)).T)
        rtree2 = KDTree(np.vstack((rt_x2, rt_y2)).T)
        # 初始化最终路径xy坐标数组
        path_x, path_y = [], []

        j1 = 0
        j2 = 0
        # 开始采样生长，判断是否找到路径或是否超过采样次数
        while self.check_end(rtree1, tree1_map, j1, rtree2, tree2_map, j2) and self.check_len(rt_x1, rt_x2):
            # 更改第一棵树的生长目标节点为第二棵树新生长出的节点
            if j2 >= 1:
                plan_gx = rt_x2[j2]
                plan_gy = rt_y2[j2]
            # 采样并找到距离采样点最近的第一棵rrt树上的节点，从该节点出发向采样点生长一段距离step
            sample_x1, sample_y1 = self.sampling(plan_gx, plan_gy, obstree)
            distance1, index1 = rtree1.query(np.array([sample_x1, sample_y1]))
            gx1 = rt_x1[index1]
            gy1 = rt_y1[index1]
            nx1 = gx1 + self.step*(sample_x1-gx1)/distance1
            ny1 = gy1 + self.step*(sample_y1-gy1)/distance1
            # 判断生长产生的新节点是否与障碍物有碰撞或是否与两棵rrt树上现有节点重复（距离过近）
            if not self.check_obs(gx1, gy1, nx1, ny1, obstree, plan_sx, plan_sy) and not self.check_rpt(nx1, nx1, rtree1, rtree2):
                # 将新节点加入第一棵rrt的KDtree
                j1 = j1 + 1
                rt_x1.append(nx1)
                rt_y1.append(ny1)
                rtree1 = KDTree(np.vstack((rt_x1, rt_y1)).T)
                # 将新节点加入第一棵rrt树图，并给新节点赋予父节点，给其父节点赋予子节点
                tree1_map.append(Node1([nx1, ny1], j1))
                tree1_map[j1].seek_parent(tree1_map[index1])
                tree1_map[index1].add_chid(tree1_map[j1])
                tree1_map[j1].cal_par_dis()

            # 采样并找到距离采样点最近的第二棵rrt树上的节点，从该节点出发向采样点生长一段距离step
            sample_x2, sample_y2 = self.sampling(rt_x1[j1], rt_y1[j1], obstree)
            distance2, index2 = rtree2.query(np.array([sample_x2, sample_y2]))
            gx2 = rt_x2[index2]
            gy2 = rt_y2[index2]
            nx2 = gx2 + self.step * (sample_x2 - gx2) / distance2
            ny2 = gy2 + self.step * (sample_y2 - gy2) / distance2
            # 判断生长产生的新节点是否与障碍物有碰撞或是否与两棵rrt树上现有节点重复（距离过近）
            if not self.check_obs(gx2, gy2, nx2, ny2, obstree, plan_gx, plan_gy) and not self.check_rpt(nx2, ny2, rtree1, rtree2):
                # 将新节点加入第二棵rrt的KDtree
                j2 = j2 + 1
                rt_x2.append(nx2)
                rt_y2.append(ny2)
                rtree2 = KDTree(np.vstack((rt_x2, rt_y2)).T)
                # 将新节点加入第二棵rrt树图，并给新节点赋予父节点，给其父节点赋予子节点
                tree2_map.append(Node2([nx2, ny2], j2))
                tree2_map[j2].seek_parent(tree2_map[index2])
                tree2_map[index2].add_chid(tree2_map[j2])
                tree2_map[j2].cal_par_dis()
        # 计算两棵rrt树上节点的最近距离及各自的索引
        distance11, index11 = rtree1.query(np.array([tree2_map[j2].data[0], tree2_map[j2].data[1]]))
        distance22, index22 = rtree2.query(np.array([tree1_map[j1].data[0], tree1_map[j1].data[1]]))
        # 初始化连通图
        road_map = []
        parent_map = []
        # 按照节点顺序将各个节点的子节点加入连通图
        for i in range(len(rtree1.data)):
            road_map.append([child.get_index() for child in tree1_map[i].children])
            rt_x.append(rt_x1[i])
            rt_y.append(rt_y1[i])
            parent_map.append(tree1_map[i].parent.index if i !=0 else -1)
        for j in range(len(rtree2.data)):
            road_map.append([child.get_index() + len(rtree1.data) for child in tree2_map[j].children])
            rt_x.append(rt_x2[j])
            rt_y.append(rt_y2[j])
            parent_map.append(tree2_map[j].parent.index + len(rtree1.data) if j !=0 else -1)

        # 回溯路径，判断是由哪棵树完成生长的最后一次而找到了路径
        if distance11 <= distance22:
            # 回溯第二棵树的路径
            path2_x, path2_y = [], []
            while j2 != 0:
                path2_x.append(tree2_map[j2].get_data()[0])
                path2_y.append(tree2_map[j2].get_data()[1])
                j2 = tree2_map[j2].parent.get_index()
                #print(j2)
            path2_x.append(tree2_map[0].get_data()[0])
            path2_y.append(tree2_map[0].get_data()[1])
            # 回溯第一棵树的路径
            path1_x, path1_y = [], []
            while index11 != 0:
                path1_x.append(tree1_map[index11].get_data()[0])
                path1_y.append(tree1_map[index11].get_data()[1])
                index11 = tree1_map[index11].parent.get_index()
            path1_x.append(tree1_map[0].get_data()[0])
            path1_y.append(tree1_map[0].get_data()[1])

            # 将两条路径拼接
            for i in range(len(path2_x)):
                path_x.append(path2_x[len(path2_x)-i-1])
                path_y.append(path2_y[len(path2_x)-i-1])

            for i in range(len(path1_x)):
                path_x.append(path1_x[i])
                path_y.append(path1_y[i])
        else:
            path2_x, path2_y = [], []
            while index22 != 0:
                path2_x.append(tree2_map[index22].get_data()[0])
                path2_y.append(tree2_map[index22].get_data()[1])
                index22 = tree2_map[index22].parent.get_index()
            path2_x.append(tree2_map[0].get_data()[0])
            path2_y.append(tree2_map[0].get_data()[1])

            path1_x, path1_y = [], []
            while j1 != 0:
                path1_x.append(tree1_map[j1].get_data()[0])
                path1_y.append(tree1_map[j1].get_data()[1])
                j1 = tree1_map[j1].parent.get_index()
            path1_x.append(tree1_map[0].get_data()[0])
            path1_y.append(tree1_map[0].get_data()[1])

            for i in range(len(path2_x)):
                path_x.append(path2_x[len(path2_x) - i - 1])
                path_y.append(path2_y[len(path2_x) - i - 1])

            for i in range(len(path1_x)):
                path_x.append(path1_x[i])
                path_y.append(path1_y[i])

        # 优化路径
        patho_x, patho_y = self.optimize(path_x, path_y, obstree)
        olen = 0
        print('how many nodes', len(patho_x))
        for i in range(len(patho_x)-1):
            olen += math.hypot(patho_x[i]-patho_x[i+1], patho_y[i]-patho_y[i+1])
            print('now len', olen)
        # return olen
        # 将优化后路径上的各节点加入optroad数组
        optroad = []
        for k in range(len(patho_x)):
            optroad.append(Node([patho_x[k], patho_y[k]], k))
        for l in range(1, len(patho_x) - 1):
            dx1 = patho_x[l] - patho_x[l - 1]
            dy1 = patho_y[l] - patho_y[l - 1]
            dx2 = patho_x[l + 1] - patho_x[l]
            dy2 = patho_y[l + 1] - patho_y[l]
            theta1 = math.atan2(dy1, dx1)
            theta2 = math.atan2(dy2, dx2)
            delta = min(math.pi - theta1 + theta2, math.pi + theta1 - theta2)
            if delta < 5 * math.pi / 6:
                optroad[l].turning = 1

        # 记录结束时间
        end_time = time.time()
        
        # 根据优化后路径上各节点与其前后节点连线的夹角值判断是否为一个转弯点
        turning = []
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

        pathi_x, pathi_y = self.inter(patho_x, patho_y)

        defult_turning = []
        for i in range(len(pathi_x)):
            defult_turning.append(0)
        pathi_x = list(reversed(pathi_x))
        pathi_y = list(reversed(pathi_y))

        patho_x = list(reversed(patho_x))
        patho_y = list(reversed(patho_y))

        # 返回优化路径，转弯点判断，规划路径（未优化），rrt树上的各点
        # return patho_x, patho_y, turning, rt_x, rt_y, parent_map
        return pathi_x, pathi_y, defult_turning

    # 采样函数，输入目标点及障碍物的KDtree，返回一个合理的采样点的xy坐标，为了提高收敛速度，采样有50%概率进行全图随机采样，有50%概率直接将目标点作为本次采样点
    def sampling(self, dir_x, dir_y, obstree):
        """

        :param dir_x: 终点横坐标
        :param dir_y: 终点纵坐标
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
                sample_x = dir_x
                sample_y = dir_y
                cc = 0

        return sample_x, sample_y

    # 检测是否找到可行路径函数，若找到则返回False，否则返回True
    def check_end(self, rtree1, tree1_map, j1, rtree2, tree2_map, j2):
        """

        :param rtree1: 第一棵rrt树的KDtree
        :param tree1_map: 第二棵rrt树的节点序列
        :param j1: 第一棵rrt树刚刚生长出的节点索引
        :param rtree2: 第二棵rrt树的KDtree
        :param tree2_map: 第二棵rrt树的节点序列
        :param j2: 第二棵rrt树刚刚生长出的节点索引
        :return: Ture or False
        """
        # 计算两棵rrt树上节点间最近距离
        distance1, index1 = rtree1.query(np.array([tree2_map[j2].data[0], tree2_map[j2].data[1]]))
        distance2, index2 = rtree2.query(np.array([tree1_map[j1].data[0], tree1_map[j1].data[1]]))
        if distance1 >= self.achieve_dis and distance2 >= self.achieve_dis:
            return True

        return False

    # 检测两点之间连线是否与障碍物发生碰撞，若有碰撞则返回True，否则返回False
    def check_obs(self, ix, iy, nx, ny, obstree, plan_sx, plan_sy):
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
        step_size = self.robot_size + self.avoid_dist
        steps = round(dis/step_size)
        # 两点间逐步检测
        for i in range(steps):
            distance, index = obstree.query(np.array([x, y]))
            if distance <= self.robot_size + self.avoid_dist and (x, y) != (plan_sx, plan_sy):
                return True
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)
        # 检测终点
        distance, index = obstree.query(np.array([nx, ny]))
        if distance <= self.robot_size + self.avoid_dist:
            return True

        return False

    # 判断新生节点是否与两棵rrt树上已有节点重复（距离小于一定阈值），如果重复则返回True，否则返回False
    def check_rpt(self, nx, ny, rtree1, rtree2):
        """

        :param nx: 节点横坐标
        :param ny: 节点纵坐标
        :param rtree1: 第一棵rrt树的KDtree
        :param rtree2: 第二棵rrt树的KDtree
        :return: True or False
        """
        distance1, index1 = rtree1.query(np.array([nx, ny]))
        distance2, index2 = rtree2.query(np.array([nx, ny]))
        if distance1 <= self.rpt_dis or distance2 <= self.rpt_dis:
            return True
        return False

    # 检查两棵rrt树总节点数是否超范围，如果超过预设上限则返回False，否则返回Tur
    def check_len(self, rt_x1, rt_x2):
        """

        :param rt_x1: 第一棵rrt树的节点横坐标序列
        :param rt_x2: 第二棵rrt树的节点横坐标序列
        :return: True or False
        """
        LEN = len(rt_x1) + len(rt_x2)
        if LEN >= self.N_SAMPLE:
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
            for j in range(i + 2, len(path_x)):
                if self.check_obs(path_x[i], path_y[i], path_x[j], path_y[j], obstree, path_x[0], path_y[0]):
                    flag = 1
                    break
            # 记录本次优化的起点索引
            sindex = i

            # 如果遍历整条路径上的点均无碰撞，则本次优化的终点即为原路径终点，否则本次优化终点索引为找到的第一个发生碰撞的节点的上一个节点的索引
            if j == len(path_x) - 1 and not flag:
                gindex = j
            else:
                gindex = j - 1
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
            if i >= len(path_x) - 2:
                break

        return path_x, path_y

    def inter(self, path_x, path_y):
        """

        :param path_x: 待插值的路径上节点的横坐标序列
        :param path_y: 待插值的路径上节点的纵坐标序列
        :return: 插值后的路径上节点的横纵坐标序列
        """
        # 初始化插值步长，新路径节点坐标序列
        lstep = 0.3
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



