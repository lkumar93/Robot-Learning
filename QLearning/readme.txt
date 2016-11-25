 THIS FOLDER CONTAINS THE IMPLEMENTATION OF THE Q-LEARNING ALGORITHM
 TO PLAY TIC TAC TOE

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

 - Install the required libraries using the following commands
	1) sudo apt-get install python-pygame
	2) sudo apt-get install python-matplotlib
	3) sudo apt-get install python-pip
	4) pip install h5py
	5) sudo apt-get install libhdf5-serial-dev
	6) sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
	7) sudo pip install Theano
	8) sudo pip install keras

 - Once you install keras, change the backend to theano as shown in the link below
	https://keras.io/backend/

 - The folder contains 4 source code files, TicTacToe.py contains the implementation of the TicTacToe game , QLearning.py 
   contains the implementation of the Q Learning Algorithm, Train.py is used to train the Qlearner for 30,000 episodes by 
   self play and Play.py can be used by the user to play with the Qlearner. The learnt policies by the qlearner have been
   saved as either a JSON file or a HDF5 file based on whether a look up table is used or function approximation is used

 - Usage :

   	Training -

		python Train.py -f <bool> -r <bool> -e <bool>

		By default, the qlearner trains itself by playing against another qlearner using Look up table

		To enable function approximation, experience replay and a random opponent , use the following command

		python Train.py -f True -r True -r True

	Testing -
		
		python Play.py -f <bool> -e <bool>

		By default, the qlearner plays against the user using Look Up Table

		To enable function approximation and experience replay , use the following command

		python Play.py -f True -e True

