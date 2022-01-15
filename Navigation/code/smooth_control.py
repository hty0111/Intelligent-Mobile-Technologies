import matplotlib.pyplot as plt
import math
from numpy import *

class SmoothCtrl(object):
    def __init__(self, current_pose, target_pose):
        self.k1 = 1
        self.k2 = 3
        self.beta = 0.4
        self.lamda = 2
        
        self.stop_distance = 0.1
        self.v_max = 1.5
        self.w_max = 5

        self.current_pose = current_pose
        self.target_pose = target_pose

        self.trajectory_predict = []
        self.control_predict = []
        self.t_predict = 0

    def feedback_vw(self, pose):
        pose_relative = self.PoseRelative(self.target_pose, pose)
        dr = math.sqrt(pose_relative[0]**2 + pose_relative[1]**2)
        if (dr<0.0001):
            return 0,0
        sigma = self.mod_angle(math.atan2(pose_relative[1], pose_relative[0]))
        theta = self.mod_angle(sigma - pose_relative[2])
        part1 = self.k2 * (-sigma - math.atan(self.k1 * theta))
        part2 = (1 + self.k1 / (1 + (self.k1 * theta)**2)) * math.sin(-sigma)
        k = (part1 + part2)/-dr
        v = self.v_max / (1 + self.beta * abs(k)**self.lamda)
        w = k * v
        return self.limit_vw(v,w)

    def predict(self, t = 0.1):
        self.trajectory_predict = []
        self.control_predict = []
        temp_pose = self.current_pose        
        for i in range(1000):            
            temp_v, temp_w = self.feedback_vw(temp_pose)
            current_x,current_y,current_angle = self.predict_onestep(temp_v, temp_w, temp_pose, t)
            self.control_predict.append([temp_v, temp_w])
            self.trajectory_predict.append([current_x,current_y,current_angle])
            temp_pose = [current_x,current_y,current_angle]
            pose_relative = self.PoseRelative(self.target_pose,[current_x,current_y,current_angle])
            if (abs(pose_relative[0])<self.stop_distance and abs(pose_relative[1])<self.stop_distance):
                break
        self.t_predict = t*len(self.trajectory_predict)
        return self.trajectory_predict,self.control_predict

    def update_current_pose(self,current_pose):
        self.current_pose = current_pose

    def update_target_pose(self,target_pose):
        self.target_pose = target_pose

    def PoseRelative(self, target_pose, current_pose):
        ca = math.cos(self.mod_angle(current_pose[2]))
        sa = math.sin(self.mod_angle(current_pose[2]))
        px = target_pose[0]-current_pose[0]
        py = target_pose[1]-current_pose[1]
        pa = target_pose[2]-current_pose[2]
        return [ca*px + sa*py, -sa*px + ca*py, self.mod_angle(pa)]

    def mod_angle(self, a):
        if a == None:
            return None
        a = a % (2 * math.pi)
        if (a >= math.pi):
            a = a - 2 * math.pi
        return a

    def limit_vw(self,v,w):
        temp_v = v
        temp_w = w
        if (w < -self.w_max):
            temp_w = -self.w_max
        if (w > self.w_max):
            temp_w = self.w_max
        if (v < -self.v_max):
            temp_v = -self.v_max
        if (v > self.v_max):
            temp_v = self.v_max
        return (temp_v,temp_w)

    def predict_onestep(self, v, w, pose, t):
        current_x = pose[0] + t*v*math.cos(pose[2])
        current_y = pose[1] + t*v*math.sin(pose[2])
        current_angle = self.mod_angle(pose[2] + t*w)
        return current_x,current_y,current_angle

class SmoothCtrlPredict(object):
    def __init__(self, current_pose, target_pose_list, dt = 0.1):
        self.current_pose = current_pose
        self.dt = 0.1
        if len(target_pose_list) == 0:
            print("target_pose_list input error")
            return
        self.target_list = []
        for i in range(len(target_pose_list)):
            if i<(len(target_pose_list)-1):
                temp_angle = self.compute_angle(target_pose_list[i],target_pose_list[i+1])
                self.target_list.append((target_pose_list[i][0],target_pose_list[i][1],temp_angle))
            else:
                if len(target_pose_list)==1:
                    temp_angle = self.compute_angle(current_pose, target_pose_list[i])
                    self.target_list.append((target_pose_list[i][0],target_pose_list[i][1],temp_angle))
                else:
                    temp_angle = self.target_list[i-1][2]
                    self.target_list.append((target_pose_list[i][0],target_pose_list[i][1],temp_angle))

    def compute_angle(self, point_1,point_2):
        # -pi ~ +pi
        return math.atan2(point_2[1]-point_1[1], point_2[0]-point_1[0])

    def feedback_vw(self, BOOST = True):
        p2p = SmoothCtrl(self.current_pose, self.target_list[0])
        trajectory_predict, control_predict = p2p.predict()
        if BOOST == True:
            trajectory_predict, control_predict = self.boost_predict(trajectory_predict, control_predict)
        if len(control_predict)==0:
            return (0,0)
        return control_predict[0]

    def predict(self, BOOST = True):
        self.trajectory_predict = []
        self.control_predict = []
        temp_pose = self.current_pose
        for i in range(len(self.target_list)):
            p2p =  SmoothCtrl(temp_pose, self.target_list[i])
            temp_trajectory_predict, temp_control_predict = p2p.predict(self.dt)
            self.trajectory_predict = self.trajectory_predict + temp_trajectory_predict
            self.control_predict = self.control_predict + temp_control_predict
            if len(temp_trajectory_predict)==0:
                continue
            else:
                temp_pose = temp_trajectory_predict[len(temp_trajectory_predict)-1]
        if BOOST:
            return self.boost_predict(self.trajectory_predict, self.control_predict)
        else:
            return self.trajectory_predict, self.control_predict

    def boost_predict(self, trajectory_predict, control_predict):
        VMAX = 1.5
        WMAX = 5
        MAX_STEP = 4
        
        self.boost_trajectory_predict = []
        self.boost_control_predict = []
        now_step_index = 0
        next_step_index = 0
        new_v = 0
        new_w = 0
        while (next_step_index <= len(control_predict)):
            if (next_step_index<len(control_predict) and abs(new_v + control_predict[next_step_index][0])<=VMAX and abs(new_w + control_predict[next_step_index][1])<=WMAX and (next_step_index-now_step_index)<MAX_STEP):
               new_v = new_v + control_predict[next_step_index][0]
               new_w = new_w + control_predict[next_step_index][1]
               next_step_index = next_step_index+1              
            else:
                self.boost_control_predict.append([new_v,new_w])
                self.boost_trajectory_predict.append(trajectory_predict[next_step_index-1])
                new_v, new_w = 0, 0
                now_step_index = next_step_index
                if next_step_index==len(control_predict):
                    break
        return self.boost_trajectory_predict, self.boost_control_predict
    
    def plot(self, BOOST = True):
        if BOOST==False:
            T = [self.dt*i for i in range(len(self.control_predict))]
            v_list = [self.control_predict[i][0] for i in range(len(self.control_predict))]
            w_list = [self.control_predict[i][1] for i in range(len(self.control_predict))]
            x_list = [self.trajectory_predict[i][0] for i in range(len(self.trajectory_predict))]
            y_list = [self.trajectory_predict[i][1] for i in range(len(self.trajectory_predict))]
        else:
            T = [self.dt*i for i in range(len(self.boost_control_predict))]
            v_list = [self.boost_control_predict[i][0] for i in range(len(self.boost_control_predict))]
            w_list = [self.boost_control_predict[i][1] for i in range(len(self.boost_control_predict))]
            x_list = [self.boost_trajectory_predict[i][0] for i in range(len(self.boost_trajectory_predict))]
            y_list = [self.boost_trajectory_predict[i][1] for i in range(len(self.boost_trajectory_predict))]
        plt.figure(1)
        plt.plot(T,v_list,'r')
        plt.plot(T,w_list,'g')
        plt.savefig("velocity")
        plt.figure(2)
        plt.plot(x_list,y_list)
        plt.savefig("trajectory")            

if __name__ == "__main":
    ## test for class SmoothCtrlPredict 
    target_pose_list = [(1,0),(1,-1),(2,-1),(2,-2),(3,-2)]
    current_pose = (0,0,0)
    dt = 0.1
    car = SmoothCtrlPredict(current_pose, target_pose_list, dt)
    # get current control
    print(car.feedback_vw(False))
    # get the whole predict
    trajectory_predict,control_predict = car.predict(False)
    # plot the trajectory
    car.plot(False)