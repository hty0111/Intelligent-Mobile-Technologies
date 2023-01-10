import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np


class Map(object):
    def __init__(self):
        super(Map, self).__init__()
        self.obstacle_x = []       # obstacles
        self.obstacle_y = []
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.mapCallback)

    def mapCallback(self, msg):
        """ 获取地图中的障碍物信息 """
        map_data = np.array(msg.data).reshape((msg.info.height, -1)).transpose()
        ox, oy = np.nonzero(map_data > 50)
        self.obstacle_x = (ox * msg.info.resolution + msg.info.origin.position.x).tolist()
        self.obstacle_y = (oy * msg.info.resolution + msg.info.origin.position.y).tolist()


