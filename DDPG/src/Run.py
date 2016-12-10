
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

import sys, getopt, os
import rospy
import time
import traceback

from DDPG import DDPG

###########################################
##
##	VARIABLES
##
###########################################

epochs = 5000

test_flag = False


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
      		opts, args = getopt.getopt(argv,"t:",["test="])

   	except getopt.GetoptError:	
      		print 'Usage: python Run.py -t <bool>'
      		sys.exit(2)
   	
	for opt, arg in opts:

     		if opt in ("-t", "--test"):

			if arg == 'True' :
				test_flag = True

			elif arg == 'False' :
				test_flag = False

			else :
				print 'Usage: python Run.py -t <bool> '
			
				sys.exit(2)

			
	#Initialize Thrust Controller
	thrust_controller = DDPG('ardrone',epsilon=1.0, hidden_layer_neurons_actor=[50,100], hidden_layer_neurons_critic=[40,80])

	print "initialized"

	rate = rospy.Rate(15)
	count = 0
		

	try :
	 	while not rospy.is_shutdown() :

			thrust_controller.run()
			rate.sleep()

			if abs(thrust_controller.state[0]) > 1.02 * abs(thrust_controller.initial_state[0]) :

				thrust_controller.reset()

			else :

				thrust_controller.update()
				thrust_controller.decrement_epsilon(200000)

				count+= 1
				print " count= "+str( count ) 

	except KeyboardInterrupt:
		raise

	except :
		print 'interrupted'

		thrust_controller.reset()
		traceback.print_exc()

		try:		
		    sys.exit(0)
		except SystemExit:
		    os._exit(0) 

	finally :

		print 'interrupted'

		thrust_controller.reset()
		traceback.print_exc()

		try:		
		    sys.exit(0)
		except SystemExit:
		    os._exit(0) 



	rospy.spin()


		

