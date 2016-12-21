
#
# THIS IS AN IMPLEMENTATION OF DEEP Q LEARNING FOR POSITION CONTROL OF QUADROTORS
#
# COPYRIGHT BELONGS TO THE AUTHOR OF THIS CODE
#
# AUTHOR : LAKSHMAN KUMAR
# AFFILIATION : UNIVERSITY OF MARYLAND, MARYLAND ROBOTICS CENTER
# EMAIL : LKUMAR93@UMD.EDU
# LINKEDIN : WWW.LINKEDIN.COM/IN/LAKSHMANKUMAR1993
#
# THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THE MIT LICENSE
# THE WORK IS PROTECTED BY COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF
# THE WORK OTHER THAN AS AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.
# 
# BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO
# BE BOUND BY THE TERMS OF THIS LICENSE. THE LICENSOR GRANTS YOU THE RIGHTS
# CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND
# CONDITIONS.#

###########################################
##
##	LIBRARIES
##
###########################################

import pygame, sys, getopt, os
import rospy
import time
import traceback
import matplotlib.pyplot as plotter

from DeepQNetwork import DQN
from rospy.exceptions import ROSException

###########################################
##
##	VARIABLES
##
###########################################

EPOCHS = 500
FPS = 40
STEPS = FPS*20
STEP_RANGE = range(0,STEPS)
EPOCH_RANGE = range(0,EPOCHS)

DQN_FLAG = False
TRAIN_FLAG = False

###########################################
##
##	MAIN FUNCTION
##
###########################################

if __name__ == '__main__':

	#Initialize ros node
	rospy.init_node('DQN', anonymous=True)

	#Parse command line arguments to check if the user wants to enable training and whether to activate dqn or normal q learning
	argv = sys.argv[1:]

   	try:
      		opts, args = getopt.getopt(argv,"d:t:",["dqn=","train="])

   	except getopt.GetoptError:	
      		print 'Usage: python Train.py -d <bool> -t <bool>'
      		sys.exit(2)
   	
	for opt, arg in opts:

     		if opt in ("-d", "--dqn"):

			if arg == 'True' :
				DQN_FLAG = True

			elif arg == 'False' :
				DQN_FLAG= False

			else :
				print 'Usage: python Train.py -f <bool> -e <bool>'
			
				sys.exit(2)

		elif opt in ("-t", "--train") :

			if arg == 'True' :
				TRAIN_FLAG = True

			elif arg == 'False' :
				TRAIN_FLAG = False

			else :
				print 'Usage: python Train.py -f <bool> -e <bool>'

				sys.exit(2)

		
	
	print "initialized"

	rate = rospy.Rate(12)
	epsilon_decay = []
	pygame.init()
	pygame.display.set_mode((20, 20))
	clock = pygame.time.Clock()
	rewards_per_episode = []
	count = 0

	#If the user wants to train
	if TRAIN_FLAG :

		#Initialize thrust controller as a DQN Agent
		thrust_controller = DQN('ardrone', param = 'Pitch', controller = 'PID', setpoint = 0.5, function_approximation = DQN_FLAG, epsilon = 1.0)

		#For n episodes and m steps keep looping
	 	for i in EPOCH_RANGE :
		
			thrust_controller.reset()
			total_reward_per_episode = 0.0			

			for j in STEP_RANGE :

				thrust_controller.play()
				rate.sleep()

				quit = False	

				total_reward_per_episode += thrust_controller.update_policy()
				thrust_controller.decrement_epsilon(STEPS*EPOCHS)

				#Reset if drone goes out of bounds
				if abs(thrust_controller.state[0]) > 1.5 * abs(thrust_controller.initial_state[0]) or  abs(thrust_controller.state[1]) > 5  :

					thrust_controller.reset()
					rate.sleep()
					break

				for event in pygame.event.get():

					if event.type == pygame.QUIT:
					    thrust_controller.reset()
					    pygame.quit(); 
					    sys.exit() 

					#For tuning PID controllers  
					if event.type == pygame.KEYDOWN and thrust.controller == 'PID':

					    if event.key == pygame.K_UP:
						thrust_controller.kp += 0.5
				
					    if event.key == pygame.K_DOWN:
						thrust_controller.kp -= 0.5

					    if event.key == pygame.K_LEFT:
						thrust_controller.kd += 0.5

					    if event.key == pygame.K_RIGHT:
						thrust_controller.kd -= 0.5

				    	    if event.key == pygame.K_q:
						thrust_controller.reset()
						time.sleep(2)
						break;

					    print "Kd Value = " + str(thrust_controller.kd)
					    print "Kp Value = " + str(thrust_controller.kp)
				count += 1


				print " Count = " + str(count) +" ,Epsilon = " + str(thrust_controller.epsilon)
	
			rewards_per_episode.append(total_reward_per_episode)
			epsilon_decay.append(thrust_controller.epsilon)

			print '\n \n \n rewards =' +str(total_reward_per_episode) + " ,epoch = "+str(i)
	

		
			clock.tick(60)

		plotter.figure()		

		plotter.plot(EPOCH_RANGE, rewards_per_episode ,'g',label='Rewards' )
		plotter.plot(EPOCH_RANGE, epsilon_decay ,'r',label='Epsilon' )
		#plotter.plot(range_of_values,drawing_percentages ,'b',label='Draws' )

		plotter.xlabel('Episode')
		plotter.ylabel('Rewards')

		plotter.legend(loc='lower right', shadow=True)

		plotter.title('Learning Curve - Thrust')

		plotter.savefig('../figures/ThrustLearningCurve.png')

		plotter.show()

	else :
		thrust_controller = DQN('ardrone', param = 'Pitch', controller = 'PID', setpoint = 0.5, function_approximation = DQN_FLAG, epsilon = -0.1)
		thrust_controller.reset()
		
		count = 0
		states = []
		while not rospy.is_shutdown():

			count = count + 1

			thrust_controller.play()
			rate.sleep()
			#thrust_controller.update_policy()

			if abs(thrust_controller.state[0]) > 2* abs(thrust_controller.initial_state[0]) or  abs(thrust_controller.state[1]) > 10  :
				thrust_controller.reset()

			states.append(thrust_controller.state[0])

			if count > 500 :
				break

		plotter.figure()		
		plotter.plot(range(0,len(states)), states ,'g',label='Error' )
		plotter.xlabel('Time')
		plotter.ylabel('Error')
		plotter.title('Pitch Learning Curve - PID ')
		plotter.savefig('../figures/PitchLearningCurvePID2.png')
		plotter.show()
		

	rospy.spin()

	


		

