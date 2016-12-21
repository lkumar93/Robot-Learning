
#
# THIS IS AN IMPLEMENTATION OF DEEP DETERMINISTIC POLICY GRADIENT ALGORITHM
# TO CONTROL A QUADCOPTER
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
# CONDITIONS.

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
import numpy

from DDPG import DDPG
from rospy.exceptions import ROSException

###########################################
##
##	VARIABLES
##
###########################################

EPOCHS = 5000
FPS = 25
STEPS = FPS*6
STEP_RANGE = range(0,STEPS)
EPOCH_RANGE = range(0,EPOCHS)

TRAIN_FLAG = True

###########################################
##
##	MAIN FUNCTION
##
###########################################

if __name__ == '__main__':

	#Initialize ros node
	rospy.init_node('QLearner', anonymous=True)

	#Parse command line arguments to check if the user wants to enable function approximation, experience replay or random opponent

	argv = sys.argv[1:]

   	try:
      		opts, args = getopt.getopt(argv,"t:",["train="])

   	except getopt.GetoptError:	
      		print 'Usage: python Run.py -t <bool>'
      		sys.exit(2)
   	
	for opt, arg in opts:

     		if opt in ("-t", "--test"):

			if arg == 'True' :
				TRAIN_FLAG = True

			elif arg == 'False' :
				TRAIN_FLAG = False

			else :
				print 'Usage: python Run.py -t <bool> '
			
				sys.exit(2)

	rate = rospy.Rate(FPS)	

	if TRAIN_FLAG :

		#Initialize Thrust Controller
		thrust_controller = DDPG(controller = "PID", epsilon = 1.0, setpoint = 0.4)	

		print "initialized"


		epsilon_decay = []
		pygame.init()
		pygame.display.set_mode((20, 20))
		clock = pygame.time.Clock()
		rewards_per_episode = []
		count = 0
		

	 	for i in EPOCH_RANGE :
		
			thrust_controller.reset()
			total_reward_per_episode = 0.0

			if i % 100 == 0 :
				thrust_controller.setpoint = numpy.random.uniform(0.5,2.0)			

			for j in STEP_RANGE :

				thrust_controller.run()
				rate.sleep()

				quit = False	

				total_reward_per_episode += thrust_controller.update()
				thrust_controller.decrement_epsilon(STEPS*EPOCHS)

				if abs(thrust_controller.state[0]) > 1.5 * abs(thrust_controller.initial_state[0]) or  abs(thrust_controller.state[1]) > 5  :

					thrust_controller.reset()
					rate.sleep()
					break

				for event in pygame.event.get():

					if event.type == pygame.QUIT:
					    thrust_controller.reset()
					    pygame.quit(); 
					    sys.exit() 

					if event.type == pygame.KEYDOWN:

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
				count += 1

				print " Count = " + str(count) +" ,Epsilon = " + str(thrust_controller.epsilon)
	
			rewards_per_episode.append(total_reward_per_episode)
			epsilon_decay.append(thrust_controller.epsilon)

			print '\n \n \n rewards =' +str(total_reward_per_episode) + " ,epoch = "+str(i)
	
			#print "Kd Value = " + str(thrust_controller.kd)
			#print "Kp Value = " + str(thrust_controller.kp)
		
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
		#Initialize Thrust Controller
		thrust_controller = DDPG(controller = "PID", epsilon=0.0,setpoint = 0.4)
		thrust_controller.reset()
		rate.sleep()


		while not rospy.is_shutdown() :

			thrust_controller.run()
			rate.sleep()
			thrust_controller.update()

			if abs(thrust_controller.state[0]) > 1.1 * abs(thrust_controller.initial_state[0]) or  abs(thrust_controller.state[1]) > 5  :
	
				thrust_controller.reset()
				rate.sleep()

	rospy.spin()
