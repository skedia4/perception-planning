# Integrated Perception and Planning for Autonomous Vehicles

For a detailed description, read our paper located in the report folder. Currently, this work is not published.

##### Localization and Mapping
<p float="left">
<img width="50%" src="/results/sequence05.png"/>
<img width="50%" src="/results/map_seq_05.png"/>
</p>

##### Planning
<p float="left">
<img width="50%" src="/results/map_seq_05_zoom_3d.png"/>
<img width="50%" src="/results/case2_cc.png"/>
<img width="50%" src="/results/case2_edt.png"/>
</p>




### Overview of the Project
- The pipeline of the Project is composed of three parts:
    1. Lidar-Inertial Odometry: [ICP (iterative closest point)+ EKF]
        - We follow the multi-sensor fusion approach to state estimation where the measurements from LiDAR and IMUs are integrated into the motion dynamics model to finally estimate the state of the vehicle. 
    2. Mapping: 
       -  The mapping takes the poses genereated from odometry as input and the raw point cloud data and generates the occupancy map and EDT for the planning module.
    3. Planning: 
       - We use trajectory optimization-based planning which penalizes energy usage and deviation to the user command. Moreover, we introduce collision avoidance into a trajectory optimization framework by means of a soft environmental constraint that comes from our mapping module. Two examples showed the effectiveness of this approach.

### How to use
Download the dataset from the [Box link](https://uofi.box.com/s/dhpv6liaab40irh6tv5386a9plpuxgcx) and unzip seq_data.zip to the project folder (./seq_data/kitti ) and run 
```sh
$ python odometry.py
```
Once we have the lidar poses, run the EKF multi-sensor fusion
```sh
$ python lidar-inertial-odometry.py
```
The predicted poses are saved as pred.npy and the evaluations of sequences 03, 05 and 07 on the ground truth can be found in the results folder. 

Copy the predicted poses pred.npy to the mapping folder run the followng command
```sh
$ python3 mapping_main.py
```
The occupancy map with EDT will be generated and saved as "seq_005_map_data_till_0-400.npz". This file will be input to the planning module. [Note: by default this file will not be generated to prevent existing file overide. Uncomment the line 95 on mapping_main.py to generate it as required]  

In order to run the planning module, please try to get the "seq_005_map_data_till_0-400.npz" by following the previous steps or download it directly from the [Box](https://tinyurl.com/yc7dyav7) and put it under the data folder. Check out the TrajOpt-OA.ipynb under the notebooks folder to test our planning module. The two cases shown in the report are listed in two cells. Simply uncomment the cell and run to check out the result.
