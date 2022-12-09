import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np



class GlobalPlanner(object):
    def __init__(self):
        super(GlobalPlanner, self).__init__()
        self.start_x = 0.0  # start
        self.start_y = 0.0
        self.goal_x = 8.0  # goal
        self.goal_y = -8.0
        self.plan_grid_size = 0.3
        self.plan_robot_radius = 0.8
        self.obstacle_x = []       # obstacles
        self.obstacle_y = []

        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.mapCallback)

    def mapCallback(self, msg):
        map_data = np.array(msg.data).reshape((msg.info.height, -1)).transpose()
        ox, oy = np.nonzero(map_data > 50)
        self.obstacle_x = (ox * msg.info.resolution + msg.info.origin.position.x).tolist()
        self.obstacle_y = (oy * msg.info.resolution + msg.info.origin.position.y).tolist()


