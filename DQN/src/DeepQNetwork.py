#
# THIS IS AN IMPLEMENTATION OF DEEP Q NETWORKS ALGORITHM
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
#

###########################################
##
##	LIBRARIES
##
###########################################
import random
import json
import time
import numpy
import rospy
import cPickle
import math
import curses	


from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop,Adam
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PointStamped
from mav_msgs.msg import RollPitchYawrateThrust
from nav_msgs.msg import Odometry

###########################################
##
##	CLASSES
##
###########################################

#Create a framework for finding the optimal winning policy using Q Learning Algorithm
class DQN:

    #Initialize the QLearner
    def __init__(self, drone = 'ARDrone' , param = 'Thrust', controller = 'AI', setpoint = 0.4, action_limits = [-0.02, 0.02], action_step_size = 0.01 , function_approximation = True, learning_rate = 0.05, discount_factor = 0.05, epsilon = 0.5, buffer_size = 10000, mini_batch_size = 40, replay = None, tau = 0.01) :

	self.drone = drone
	self.param = param
	self.controller = controller
	self.learning_rate = learning_rate
	self.discount_factor = discount_factor
	self.setpoint = setpoint
	self.epsilon = epsilon
	self.function_approximation = function_approximation
	self.replay_buffer = []
	self.buffer_size = buffer_size
	self.mini_batch_size = mini_batch_size
	self.initialized = False
	self.initial_state = (0.0,0.0)
	self.current_state = (0.0,0.0)
	self.state = (0.0,0.0)
	self.min_value = action_limits[0]
	self.max_value = action_limits[1]
	self.step_size = action_step_size
	self.actions = list(numpy.arange(self.min_value,self.max_value,self.step_size))
	self.actions = random.sample(self.actions,len(self.actions))
	self.reset_flag = False
	self.epochs = 0 
	self.cmd = RollPitchYawrateThrust()

	cmd_topic = '/'+ drone+'/command/roll_pitch_yawrate_thrust'
	sub_topic = '/'+ drone+'/ground_truth/position'
	self.cmd_publisher = rospy.Publisher(cmd_topic, RollPitchYawrateThrust, queue_size = 1)
	self.state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 1)
	self.tau = tau
	self.prev_z = 0.0
	self.current_z = 0.0

	if self.controller == 'PID' :
		
		if self.param == 'Thrust':
			self.kp = 1.0 #2 when loop rate 25
			self.kd = 14.5 #81.5 when loop rate 25

		elif self.param == 'Pitch':
			self.kp = 2
			self.kd = 15.5

		elif self.param == 'Roll':
			self.kp = 1
			self.kd = 1
	


	rospy.Subscriber(sub_topic, PointStamped, self.get_state)

	if replay is None :

		self.replay = function_approximation

	else :
		self.replay = replay

	self.replay_count = 0

	if function_approximation :

		self.file_name = '../policies/' + drone + '_' + param + '_' + str(self.min_value) + '_' + str(self.max_value) + '_' + str(self.step_size) + '_' + controller +'_policy_fa.h5'

		self.target_file_name = '../policies/' + drone + '_' + param + '_' + str(self.min_value) + '_' + str(self.max_value) + '_' + str(self.step_size) + '_' + controller +'target_policy_fa.h5'

	else :

		self.file_name = '../policies/' + drone + '_' + param + '_' + str(self.min_value) + '_' + str(self.max_value) + '_' + str(self.step_size) + '_' + controller +'_policy_lt.p'

	self.load_policy()
	


    #Take an action
    def play(self) :

	current_state = self.state
	current_action = None
	current_value = 0.0
	current_count = 0.0
	
	if self.function_approximation :
	
		if random.random() > self.epsilon :

			current_action, current_value, current_q_values, current_action_index= self.get_best_action_value(current_state, self.actions)

		else :

			current_action, current_value, current_q_values, current_action_index = self.get_best_action_value(current_state, self.actions, random_action = True)		

		self.current_q_values = current_q_values
		self.current_action_index = current_action_index

		#print "current_q_values = " + str(current_q_values)


	else :

		#If current state is in Q Learner's policy 
		if current_state in self.policy :

			#Epsilon-Greedy Strategy : Exploitation vs Exploration
			#Select the best action with (1-epsilon) probability
			if random.random() > self.epsilon :
				#Find the best action and value
				current_action, current_value,current_count = self.get_best_action_value(current_state,self.actions)

	 	#As current state is not in the policy, initialize a list for that particular state
		else :
			self.policy[current_state] = []

		#If the current state is not in the policy or if selecting a random action with epsilon probability 
		#Make sure you are selecting an action that has not been executed before , if all the actions have been 
		#executed before then do a random action

		if current_action is None :

			current_action, current_value,current_count = self.get_best_action_value(current_state,self.actions,random_action = True)
					
	print "current_state = " + str(current_state)

	#Execute the current action
	self.execute_action(current_action)

	self.current_state = current_state
	self.current_action = current_action
	self.current_value = current_value
	self.current_count = current_count



    #Update the Q(s,a) table	
    def update_policy(self) :

	#Find the next state after the second player has played
	next_state = self.state

	print "next_state = " + str(next_state)

	#Initialize the value for next state action pair
	next_value = 0

	next_q_values = None

	reward = self.get_reward(next_state)

	#Check which moves are legal	

	if self.function_approximation :

		print "action = " + str(self.current_action) + " value = " + str(self.current_value)


		if self.replay :

			#Store the information in a batch and randomly sample a mini batch from this and keep replaying it 
			#While updating both the policy and mini batch

			self.experience_replay(self.current_state, self.current_action, reward , next_state, self.actions, self.current_action_index)


		else :

			#Get the q values for next state

			next_action, next_value, next_q_values,next_action_index = self.get_best_action_value(next_state,self.actions)

			target = numpy.zeros( ( 1, len(self.current_q_values) ) )
		
			target[:] = self.current_q_values[:]

			if reward == 1.0 :
				target[0][self.current_action_index] = reward

			else :
				target[0][self.current_action_index] += (self.learning_rate)*(reward+ self.discount_factor*next_value - target[0][self.current_action_index])

			#Train the function approximator to fit this data

			self.function_approximator.fit(self.current_state.reshape( 1,len(self.current_state)), target , batch_size = 1, nb_epoch = 1, verbose = 1) 



	else :
		

		#If the next state is in QLearner's policy, then
		if next_state in self.policy : 


			#Find the next action and the value of next state/action pair
			next_action, next_value,next_count = self.get_best_action_value(next_state,self.actions)

		#As next state is not in the policy, initialize a list for that particular state
		else :
			self.policy[next_state] = []

		if abs(next_state[0]) <= 0.03 and abs(next_state[1]) <= 0.005 :

			self.current_value = 10.0

		else :

			#Q(s,a) = Q(s,a) + Alpha * ( R' + Gamma*Q(s',a') - Q(s,a) )
			self.current_value = self.current_value + (self.learning_rate)*(reward+ self.discount_factor*next_value - self.current_value)
			#self.current_value = reward
			#self.current_value = (self.get_reward(next_state)+ self.discount_factor*next_value)

		#Round off to four decimal points
		self.current_value = round(self.current_value,4)
	
		action_in_policy = False

		print "action = " + str(self.current_action) + " value = " + str(self.current_value)+ "count = " + str(self.current_count)

		#If the action is in the policy for current state modify the value of the action
		for av in self.policy[self.current_state] :
			if av['action'] == self.current_action :
				action_in_policy = True
				av['value'] = self.current_value
				av['count'] = self.current_count
	
		#If the action is not in the policy, augment the action value pair to that state
		if action_in_policy is not True :
			self.policy[self.current_state].append({'action':self.current_action,'value':self.current_value,'count':self.current_count})


	self.epochs += 1

	#Save policy once in every 10000 episodes
	if self.epochs % 1000 == 0 :
		#Save the updated policy
		self.save_policy()	

	return reward
	


    def experience_replay(self, state, action, reward, new_state, actions, action_index) :
	
	#If the buffer is still not filled up, keep adding information to the buffer
	if len(self.replay_buffer) < self.buffer_size :

		self.replay_buffer.append((state, action, reward, new_state, actions, action_index))

	else:
		#After the buffer is filled, keep replacing old information with new information
		if self.replay_count < self.buffer_size - 1 :
			self.replay_count += 1
		
		else :
			self.replay_count = 0

		self.replay_buffer[self.replay_count] = (state, action, reward, new_state, actions, action_index)

	#Randomly sample a mini batch from the replay buffer

	if len(self.replay_buffer) < self.mini_batch_size :

		mini_batch = random.sample(self.replay_buffer,len(self.replay_buffer))

	else :

		mini_batch = random.sample(self.replay_buffer,self.mini_batch_size)

	training_input = []

	training_output = []


	#For all the tuples in the mini batch , update the current value based on the reward and append the target value to training output 
	#and append the current value to the target input
	reward_flag = 0
	current_state = (0.0,0.0)
	current_action = 0.0
	updated_value = 0.0
	for memory in mini_batch :

		current_state, current_action, immediate_reward, next_state, possible_actions, current_action_index = memory

		current_q_values = self.function_approximator.predict(current_state.reshape(1,len(state)),batch_size = 1)

		#Use the frozen target Q network for getting the next value
		next_action, next_value, next_q_values, next_action_index = self.get_best_action_value(next_state, possible_actions)#, use_target = True)

		target = numpy.zeros((1,len(current_q_values[0])))

		target[:] = current_q_values[:]

		#if immediate_reward >= 0.95 or immediate_reward == -1 :
		#	updated_value = immediate_reward

		#else :
		if immediate_reward == 1 or immediate_reward == -1 :
			target[0][current_action_index] = immediate_reward
		else :
			target[0][current_action_index] += (self.learning_rate)*(immediate_reward + self.discount_factor*next_value - target[0][current_action_index])

		#target[0][current_action_index] = immediate_reward

		training_input.append(current_state.reshape(len(state),))

		training_output.append(target.reshape(len(current_q_values[0]),))

	print "current_state = " + str(current_state) + " current_action = " + str(current_action) + ' updated_value'+ str(updated_value) 
	
#	print "\n \n \n current_q_values \n"
#	print current_q_values
#	print "\n target_q_values \n "
#	print target
#	print '\n'


	training_input = numpy.array(training_input)

	training_output = numpy.array(training_output)

	#Train the function approximator to fit the mini_batch

	self.function_approximator.fit(training_input, training_output, batch_size = self.mini_batch_size, nb_epoch = 1, verbose = 1)

	#self.train_target_q_network()


    #Find the best action for a particular state
    def get_best_action_value(self, state, possible_actions, random_action = False, use_target = False):

	action = None
	value = 0.0
	count = 1.0
	q_values = None
	index = 0

	if self.function_approximation :

		#Decide whether to use the frozen target q network or the updated function approximator
		if use_target :
			q_values = self.target_q_network.predict(state.reshape(1,len(state)),batch_size = 1)

		else :
			q_values = self.function_approximator.predict(state.reshape(1,len(state)),batch_size = 1)


		if random_action :

			if self.controller == 'PID' :

				PID = (self.kp * self.current_state[0]) + (self.kd* self.current_state[1])
				PID_limited =  max(min(PID,self.max_value) , self.min_value)
				index, action = min(enumerate(possible_actions), key=lambda x: abs(x[1]-PID_limited))
				print " PID = " + str(PID_limited) + " ,action = " +str(action)+' ,index = ' +str(index)
	
			else :
			
				index = random.randint(0,len(possible_actions)-1)
				action = possible_actions[index]
			
			value = q_values[0][index]

		else :
		
			#Find a legal action that has the highest value
			sorted_action_indices = numpy.argsort(q_values[0])[::-1]
			sorted_values = numpy.sort(q_values[0])[::-1]
	

			#for selected_action_index in sorted_action_indices :

			index = sorted_action_indices[0]
			action = possible_actions[index]
			value = q_values[0][index]

#			print " \n sorted_values = "
#			print sorted_values
#			print " \n sorted_action_indices = "
#			print sorted_action_indices
#			print "\n largest_action = " + str(action)
#			print "\n largest_index = " + str(index)
#			print "\n largest_value = " + str(value)
#			print "\n possible_action = \n" + str(possible_actions)


#		if action is None :
#			index = random.randint(0,len(possible_actions)-1)
#			action = possible_actions[index]			
#			value = q_values[0][index]
	

		return action, value, q_values, index

	else :
	
		#if selecting a random action with epsilon probability, make sure you are selecting an action that has been never been executed before 			
		# If all the actions have been executed before, select action with less count, if count is more than 5 times , then do a random action	
		if random_action :

			if self.controller == 'PID' :

				PID = (self.kp * self.current_state[0]) + (self.kd* self.current_state[1])
				PID_limited =  max(min(PID,self.max_value) , self.min_value)
				index, action = min(enumerate(possible_actions), key=lambda x: abs(x[1]-PID_limited))

				action_in_policy = False

				for av in self.policy[state] :

					if av['action'] == action :					
						action_in_policy = True
						value = av['value']
						count = av['count']
						break;

				if action_in_policy == False :
					
					value = 0.0
					count = 1.0

	
			else :

				action = random.choice(possible_actions)

				if len(self.policy[state]) < len(possible_actions) :
					for move in possible_actions:

						action_in_policy = False

						for av in self.policy[state] :

							if av['action'] == move :
						
								action_in_policy = True
						
							if av['action'] == action :

								value = av['value']
								count = av['count']

		
						if action_in_policy is False :
						
							action = move
							value = 0.0
							count = 1.0
							break

				else :

					sorted_av_table = []

					#Sort the actions according to the count
					sorted_av_table = sorted(self.policy[state], reverse=False, key = lambda av : av['count'])
	
					print sorted_av_table

					for av in sorted_av_table :
			
						if av['action'] == action :

							value = av['value']
							count = av['count'] + 1

						if av['count'] < 10 and av['action'] in possible_actions:
							action = av['action']
							value = av['value']
							count = av['count'] + 1
							break


		else :
		
			sorted_av_table = []

			#Sort the actions according to the values
			sorted_av_table = sorted(self.policy[state], reverse=True, key = lambda av : av['value'])

			print sorted_av_table
			for av in sorted_av_table :
				if av['value'] > -20*abs(self.initial_state[0]) and av['action'] in possible_actions :
					action = av['action']
					value = av['value']
					count = av['count'] + 1
					break

		return action, value, count


    #Initialize Neural Network
    def initialize_neural_network(self) :

	#Create a sequential neural network with 2 hidden layers of 27 and 18 neurons respectively, the output layer
	#consists of 9 neurons which maps to box numbers in the Tic Tac Toe Board

	model = Sequential()

	model.add(Dense(100, init='lecun_uniform', input_shape=(2,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(200, init='lecun_uniform'))
	model.add(Activation('relu'))
	
	model.add(Dense(len(self.actions), init='lecun_uniform'))
	model.add(Activation('linear'))

	rms = RMSprop()
	#adam = Adam(lr=0.001)
	model.compile(loss='mse', optimizer=rms)

	return model


    #Reward Function
    def get_reward(self,state) :

	#reward = math.exp(-abs(state[0])/0.2)* math.exp(-abs(state[1])/0.3)
	
	if self.function_approximation :
	
		reward = math.exp(-abs(state[0])/0.2)* math.exp(-abs(state[1])/0.3)

#		if abs(state[0]) >= abs(self.initial_state[0]) or abs(state[1]) >= 5 :
#			reward = -1.0

#		elif abs(state[0]) <= 0.05 and abs(state[1]) <= 0.005 :
#			reward = 1.0
#	
#		else :
#			reward = 0

	else :
#		if abs(state[0]) <= 0.05 and abs(state[1]) <= 0.001 :
#			reward = 1

#		elif abs(state[0]) >= abs(self.initial_state[0]) or abs(state[1]) >= 5 :
#			reward = -1

#		else :
#			reward = 0

		reward = math.exp(-abs(state[0])/0.2)* math.exp(-abs(state[1])/0.3)
	
	print reward

	return reward

    #Get state of the game
    def get_state(self, msg) :

	if self.function_approximation :

		if self.param == 'Roll' :
			value = msg.point.y
		elif self.param == 'Pitch' :
			value = msg.point.x
		else :
			value = msg.point.z

		state = round(self.setpoint - value, 4)
		self.state = numpy.array((state, round(state - self.current_state[0],4)))


		if self.param != 'Thrust':
			
			error_z = 0.6 - msg.point.z
			derror_z = error_z - self.prev_z
			self.current_z = error_z
			self.cmd.thrust.z = 15.0 + 1.0*error_z + 12.0*derror_z
	

		if self.initialized is False :
			self.initial_state = numpy.array((state, 0.0))
			self.state = self.initial_state
			self.prev_z = 0.0
			self.initialized = True

	else :

		if self.param == 'Roll' :
			value = msg.point.y
		elif self.param == 'Pitch' :
			value = msg.point.x
		else :
			value = msg.point.z

		state = round(self.setpoint - value,3 )
		self.state = (state, round(state - self.current_state[0],3))

		if self.param != 'Thrust':
			
			error_z = 0.6 - msg.point.z
			derror_z = error_z - self.prev_z
			self.current_z = error_z
			self.cmd.thrust.z = 15.0 + 1.0*error_z + 12.0*derror_z

		if self.initialized is False :
			self.initial_state = numpy.array((state, 0.0))
			self.state = self.initial_state
			self.prev_z = 0.0
			self.initialized = True


    def execute_action(self,action) :

	self.cmd.header.stamp = rospy.Time.now()

	if self.param == 'Pitch':

		self.cmd.pitch = action

		print "Pitch = " + str(self.cmd.pitch) 

	elif self.param == 'Roll' :

		self.cmd.roll = action

		print "Roll = " + str(self.cmd.roll) 

	else :

		self.cmd.thrust.z =  15.0 + action

		print "Thrust = " + str(self.cmd.thrust.z) 

	self.prev_z = self.current_z
	self.cmd_publisher.publish(self.cmd)	
	

    def decrement_epsilon(self,value) :

	if self.epsilon > 0.0 :
		self.epsilon -= 1.0/value
	else :
		self.epsilon = 0.0

    def train_target_q_network(self) :

	fa_weights = self.function_approximator.get_weights()
	target_weights = self.target_q_network.get_weights()

	for i in xrange(len(fa_weights)) :
		target_weights[i] = self.tau * fa_weights[i] + (1-self.tau)*target_weights[i]

	self.target_q_network.set_weights(target_weights)

    #Load the policy 
    def load_policy(self) :

	if self.function_approximation :
		#Load the neural networks param from the file if it exists
		try :
			self.function_approximator = load_model(self.file_name)
			self.target_q_network = load_model(self.target_file_name)
		
		#Else initialize the neural network			
		except IOError:
		
			self.function_approximator = self.initialize_neural_network()
			self.target_q_network = self.initialize_neural_network()
			print 'Initialized Neural Network at ' + self.file_name + ' and ' + self.target_file_name
			return				
		
		print self.file_name + ' loaded'
		print self.target_file_name + ' loaded'


	else :

		self.policy = {}

		#Open the json file if available
		try:
			policy_file = open(self.file_name, 'rb')
		except IOError:
			print "No such file exists. Will create one"
			time.sleep(3.0)
			return

		#Load the contents of the file to QLearner's policy
		self.policy = cPickle.load(policy_file)

		#Close the policy
		policy_file.close()

		print self.file_name + ' loaded'

    #Save the policy
    def save_policy(self) :

	if self.function_approximation :

		#Save the architecture and weights of neural network as a HDF5 file
		self.function_approximator.save(self.file_name)
		self.target_q_network.save(self.target_file_name)

		print self.target_file_name + ' saved'	

	else :
		#Save the policy as a json file
		policy_file = open(self.file_name,'wv')
	
		#Dump the dictionary in QLearner's file as a Pickle Object to policy file
		cPickle.dump(self.policy,policy_file) #,sort_keys = True,indent = 4, separators=(',',': '))

		#Close the policy file
		policy_file.close()

	print self.file_name + ' saved'	

    def reset(self) :


	cmd = RollPitchYawrateThrust()
	cmd.header.stamp = rospy.Time.now()
	cmd.roll = 0.0
	cmd.pitch = 0.0
	cmd.thrust.z = 0.0
	self.reset_flag = True
	self.cmd_publisher.publish(cmd)	

	time.sleep(0.2)
		
	reset_cmd = ModelState()
	reset_cmd.model_name = self.drone
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

	self.state_publisher.publish(reset_cmd)

	self.cmd_vel = 0.0
	self.initialized = False
	#self.current_state = self.initial_state
	self.reset_flag = False

	self.prev_z = 0.0
	self.current_z = 0.0

	time.sleep(0.2)
	

	
