 THIS FOLDER CONTAINS THE IMPLEMENTATION OF THE DEEP Q NETWORKS
 TO CONTROL THE POSITION OF QUADROTORS

 COPYRIGHT BELONGS TO THE AUTHOR OF THIS REPOSITORY

 AUTHOR : LAKSHMAN KUMAR
 AFFILIATION : UNIVERSITY OF MARYLAND, MARYLAND ROBOTICS CENTER
 EMAIL : LKUMAR93@UMD.EDU
 LINKEDIN : WWW.LINKEDIN.COM/IN/LAKSHMANKUMAR1993

 THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THE MIT LICENSE
 THE WORK IS PROTECTED BY COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF
 THE WORK OTHER THAN AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.
 
 BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO
 BE BOUND BY THE TERMS OF THIS LICENSE. THE LICENSOR GRANTS YOU THE RIGHTS
 CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND
 CONDITIONS.

 INSTRUCTIONS

 - Ubuntu 14.04 and Python 2.7+ are required to use this repository

 - Install the required libraries using the following commands and set packages using the following commands
	1) sudo apt-get install python-pygame
	2) sudo apt-get install python-matplotlib
	3) sudo apt-get install python-pip
	4) pip install h5py
	5) sudo apt-get install libhdf5-serial-dev
	6) sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
	7) sudo pip install Theano
	8) sudo pip install keras
 	9) sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'
	10) wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
	11) sudo apt-get update
	12) sudo apt-get install ros-indigo-desktop-full ros-indigo-joy ros-indigo-octomap-ros python-wstool python-catkin-tools
	13) sudo apt-get install ros-indigo-gazebo-*
	14) sudo rosdep init
	15) rosdep update
	16) source /opt/ros/indigo/setup.bash
	17) mkdir -p ~/catkin_ws/src
	18) cd ~/catkin_ws/src
	19) catkin_init_workspace
	20) wstool init
	21) git clone https://github.com/ethz-asl/rotors_simulator.git
	22) git clone https://github.com/ethz-asl/mav_comm.git
	23) cd ~/catkin_ws
	24) catkin init
	25) catkin build
	26) echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
	27) source ~/.bashrc

 - Once you install keras, change the backend to theano as shown in the link below
	https://keras.io/backend/

 - The folder contains 2 source code files, DQN.py contains the implementation of the DQN & Q Learning algorithm , Train.py is used to train   
   the Quadrotor for 500 episodes by self play . The learnt policies by the qlearner have been saved as either a cPickle file or a HDF5 file   
   based on whether a look up table is used or DQN is used

 - Before running the reinforcement learning module, do 
   roslaunch rotors_gazebo mav_with_joy.launch mav_name:=ardrone world_name:=basic 

   This will launch gazebo and click the play button on the gazebo gui to enable physics

-  After this you can run the modules as follows

 - Usage :

   	Training -

		python Train.py -d <bool> -t <bool> 

		By default, the qlearner trains itself by using Look up table

		To enable DQN , use the following command

		python Train.py -d True -t True 

	Testing -
		
		python Train.py -d <bool> -t False

		By default, the qlearner controls quadrotor using Look Up Table and trains

		To enable DQN and disable training , use the following command

		python Train.py -d True -t False

	In order to control Roll, Pitch or Thrust, specify that option when initializing the DQN agent in Train.py

