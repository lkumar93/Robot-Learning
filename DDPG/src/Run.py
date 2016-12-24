
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

EPOCHS = 50000
FPS = 10
STEPS = FPS*10
STEP_RANGE = range(0,STEPS)
EPOCH_RANGE = range(0,EPOCHS)
DRONE = 'ardrone'
TUNEPARAM = 'Thrust'

DQN_FLAG = True
TRAIN_FLAG = True

###########################################
##
##	HELPER FUNCTIONS
##
###########################################

def init_controllers() :

	thrust_controller = DDPG(DRONE, param = 'Thrust', controller = 'PID' , setpoint = 0.5, epsilon = 1.0)
	roll_controller = DDPG(DRONE, param = 'Roll', controller = 'PID', setpoint = 0.5,epsilon = 1.0)	
	pitch_controller = DDPG(DRONE, param = 'Pitch', controller = 'PID', setpoint = 0.5, epsilon = 1.0)

	return [thrust_controller,roll_controller,pitch_controller]

def run(controllers, cmd_publisher) :

	cmd = RollPitchYawrateThrust()

	cmd.header.stamp = rospy.Time.now()

	for controller in controllers:

		if controller.param == 'Thrust' :

			cmd.thrust.z = 15.0 + controller.run()
			print "Thrust = " + str(cmd.thrust.z)

		elif controller.param == 'Roll' :

			cmd.roll = controller.run()
			print "Roll = " + str(cmd.roll)

		elif controller.param == 'Pitch' :

			cmd.pitch = controller.run()
			print "Pitch = " + str(cmd.pitch)

	cmd_publisher.publish(cmd)

def epsilon_decay(controllers) :
	
	for controller in controllers :

		controller.decrement_epsilon(STEPS*EPOCHS)


def update(controllers) :
	
	reward = 0.0

	for controller in controllers:

		reward+=controller.update()

	return reward


def tune_pid(controllers, param, val, coeff) :
	
	for controller in controllers :
	
		if controller.param == param and controller.controller == 'PID':

			if coeff == 'kp' :

				controller.kp += val

			elif coeff == 'kd' :
				
				controller.kd += val

		        print param+" Kd Value = " + str(controller.kd)
		        print param+" Kp Value = " + str(controller.kp)

def check_validity(controllers) :

	for controller in controllers :

		if controller.check_validity() is False :

			return False

	return True

def extract_state(controllers,param) :

	for controller in controllers :

		if controller.param == param :

			return controller.state[0]


def simulator_reset(controllers,cmd_publisher,gazebo_publisher) :


	cmd = RollPitchYawrateThrust()
	cmd.header.stamp = rospy.Time.now()
	cmd.roll = 0.0
	cmd.pitch = 0.0
	cmd.thrust.z = 0.0

	cmd_publisher.publish(cmd)

	time.sleep(0.1)

	reset_cmd = ModelState()
	reset_cmd.model_name = DRONE
	reset_cmd.pose.position.x = 0.0
	reset_cmd.pose.position.y = 0.0
	reset_cmd.pose.position.z = 0.07999
	reset_cmd.pose.orientation.x = 0.0
	reset_cmd.pose.orientation.y = 0.0
	reset_cmd.pose.orientation.z = 0.0
	reset_cmd.pose.orientation.w = 1.0
	reset_cmd.twist.linear.x = 0.0
	reset_cmd.twist.linear.y = 0.0
	reset_cmd.twist.linear.z = 0.0
	reset_cmd.twist.angular.x = 0.0
	reset_cmd.twist.angular.y = 0.0
	reset_cmd.twist.angular.z = 0.0
	reset_cmd.reference_frame= 'world'

	gazebo_publisher.publish(reset_cmd)

	time.sleep(0.05)

	for controller in controllers :
		controller.reset()

	time.sleep(0.05)




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

	
	cmd_topic = '/'+ DRONE+'/command/roll_pitch_yawrate_thrust'

	cmd_publisher = rospy.Publisher(cmd_topic, RollPitchYawrateThrust, queue_size = 1)
	gazebo_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 1)

	rate = rospy.Rate(FPS)

	pygame.init()
	pygame.display.set_mode((20, 20))
	clock = pygame.time.Clock()
	rewards_per_episode = []
	count = 0

	#Initialize position controllers as DDPG Agents
	controllers = init_controllers()

	print "initialized"

	#If the user wants to train
	if TRAIN_FLAG :

		
		#For n episodes and m steps keep looping
	 	for i in EPOCH_RANGE :
		
			simulator_reset(controllers,cmd_publisher,gazebo_publisher)
			total_reward_per_episode = 0.0			

			for j in STEP_RANGE :

				run(controllers,cmd_publisher)

				rate.sleep()

				total_reward_per_episode += update(controllers)
				epsilon_decay(controllers)

				#Reset if drone goes out of bounds
				if check_validity(controllers) is False :

					simulator_reset(controllers, cmd_publisher, gazebo_publisher)
					rate.sleep()
					break

				for event in pygame.event.get():

					if event.type == pygame.QUIT:

					    simulator_reset(controllers, cmd_publisher, gazebo_publisher)
					    pygame.quit(); 
					    sys.exit() 

					#For tuning PID controllers  
					if event.type == pygame.KEYDOWN :

					    if event.key == pygame.K_UP:
						tune_pid(controllers, TUNEPARAM , 0.5, 'kp') 
				
					    if event.key == pygame.K_DOWN:
						tune_pid(controllers, TUNEPARAM , -0.5, 'kp') 

					    if event.key == pygame.K_LEFT:
						tune_pid(controllers, TUNEPARAM, 0.5, 'kd') 

					    if event.key == pygame.K_RIGHT:
						tune_pid(controllers, TUNEPARAM , -0.5, 'kd') 

				    	    if event.key == pygame.K_q:
						simulator_reset(controllers,cmd_publisher,gazebo_publisher)
						time.sleep(2)
						break;

				count += 1


				print " Count = " + str(count) +" ,Epsilon = " + str(controllers[0].epsilon)
	
			rewards_per_episode.append(total_reward_per_episode)
	
			print '\n \n \n rewards =' +str(total_reward_per_episode) + " ,epoch = "+str(i)	

		
			clock.tick(FPS*2)

		plotter.figure()		

		plotter.plot(EPOCH_RANGE, rewards_per_episode,'g',label='Rewards' )

		plotter.xlabel('Episode')
		plotter.ylabel('Rewards')

		plotter.title('Learning Curve ')

		plotter.savefig('../figures/LearningCurve.png')

		plotter.show()

	else :

		simulator_reset(controllers,cmd_publisher,gazebo_publisher)
		
		count = 0
		states = []
		while not rospy.is_shutdown():

			count = count + 1
			run(controllers,cmd_publisher)
			rate.sleep()
			
			if check_validity(controllers) is False :
				simulator_reset(controllers,cmd_publisher,gazebo_publisher)

			states.append(extract_state(controllers,'Thrust'))

			if count > 500 :
				break

		plotter.figure()		
		plotter.plot(range(0,len(states)), states ,'g',label='Error' )
		plotter.xlabel('Time')
		plotter.ylabel('Error')
		plotter.title('Learning Curve - PID ')
		plotter.savefig('../figures/LearningCurvePID.png')
		plotter.show()
		

	rospy.spin()

