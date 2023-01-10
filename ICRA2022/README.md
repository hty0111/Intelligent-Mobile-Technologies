# 使用说明

## 1. 工程目录

|— `src/` 

​	|— `scripts/`

​		|— `run.py`		  // 运行文件	

​		|— `gazebo_simulation.py`		// gazebo仿真相关，未修改

​		|— `a_star.py`	// 路径规划：A*

​		|— `dwa.py`		  // 轨迹规划：DWA

​		|— `map.py`		  // 用于读取地图信息（障碍物信息）

​		|— `rrt.py` / `rrt_star.py` / `rrt_connect` / `rrt_star_connect`	// 其他的路径规划算法：RRT、RRT*、双向RRT、双向RRT *



## 2. 环境配置

​		运行环境： `Ubuntu18.04`, `python3`, `ROS melodic`



## 3. 编译运行

```shell
catkin_make
python3 src/scripts/run.py --world_idx xxx --gui	# 默认使用A*算法作为路径规划
```



## 4. 更改算法

​		在 `run.py` 文件开头进行更改，选择一个路径规划算法。

```python
from a_star import Astar as Planner
# from rrt import RRT as Planner
# from rrt_star import RRTSTAR as Planner
# from rrt_connect import RRTCONNECT as Planner
# from rrt_star_connect import RRTSTANECT as Planner
```



## 5. 其他

- 如果出现gazebo中车辆模型在天上的情况，重新运行即可。
- dwa参数是根据自己的笔记本电脑调整的，real time factor 仅0.1左右，可能在其他电脑上运行时需要重新调整参数。