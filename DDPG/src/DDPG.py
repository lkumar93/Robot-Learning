#
# THIS IS AN IMPLEMENTATION OF DEEP DETERMINISTIC GRADIENT POLICY ALGORITHM
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

from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation, merge
from keras.initializations import normal, identity
from keras.optimizers import RMSprop,Adam
from keras.engine.training import collect_trainable_weights
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import PointStamped
from mav_msgs.msg import RollPitchYawrateThrust
import tensorflow as tf
import keras.backend as kb

###########################################
##
##	CLASSES
##
###########################################
class Actor :
	
	def __init__(self,tf_session,state_dim,action_dim,file_name,target_file_name,learning_rate = 1,tau = 0.001, hidden_layer_neurons = [300,600]) :
		
		self.tf_session = tf_session
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.learning_rate = learning_rate
		self.hidden_layer_neurons = hidden_layer_neurons
		self.tau = tau	

		kb.set_session(tf_session)		
	
		self.state = Input(shape=[state_dim])
		self.target = self.state
	
		self.file_name = file_name #'actor_network.h5'
		self.target_file_name = target_file_name #'actor_target_network.h5'

		self.load_network()

		self.action_gradients = tf.placeholder(tf.float32,[None,action_dim])
		self.gradient_params = tf.gradients(self.network.output, self.weights, -self.action_gradients)
		gradients = zip(self.gradient_params,self.weights)

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(gradients)

		self.tf_session.run(tf.initialize_all_variables())



	def train(self,states,action_gradients) :

		self.tf_session.run(self.optimizer, feed_dict={self.state: states, self.action_gradients: action_gradients})

	
	def train_target_network(self) :

		network_weights = self.network.get_weights()
		target_weights = self.target_network.get_weights()

		for i in xrange(len(network_weights)) :
			target_weights[i] = self.tau*network_weights[i] + (1-self.tau)*target_weights[i]

		self.target_network.set_weights(target_weights)



	def create_actor_network(self) :
		
		hidden_layer_1 =  Dense(self.hidden_layer_neurons[0], activation='relu')(self.state)
		hidden_layer_2 =  Dense(self.hidden_layer_neurons[1], activation='relu')(hidden_layer_1)

		#thrust
		output_layer = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(hidden_layer_2)   
		network = Model(input=self.state,output=output_layer)
		
		adam = Adam(lr=self.learning_rate)

		network.compile(loss='mse',optimizer=adam)
	
		return network

	def load_network(self) :

		#Load the neural networks param from the file if it exists
		
	
		self.network = self.create_actor_network()
		self.target_network = self.create_actor_network()
		
		try :
			self.network.load_weights(self.file_name)	

			self.target_network.load_weights(self.target_file_name)

			print self.file_name + ' loaded'
			print self.target_file_name + ' loaded'

		except IOError:
			
			print 'Initialized Actor Networks at ' + self.file_name + ' and ' + self.target_file_name

		self.weights = self.network.trainable_weights
		self.target_weights = self.target_network.trainable_weights		
		

	

	def save_network(self) :

		self.network.save_weights(self.file_name)
		self.target_network.save_weights(self.target_file_name)

		print self.file_name + ' saved'
		print self.target_file_name + ' saved'	





class Critic :
	
	def __init__(self,tf_session,state_dim,action_dim,file_name,target_file_name,learning_rate = 1,tau = 0.001, hidden_layer_neurons = [300,600]) :
		
		self.tf_session = tf_session
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.learning_rate = learning_rate
		self.hidden_layer_neurons = hidden_layer_neurons
		self.tau = tau
		
		self.state = Input(shape=[self.state_dim])
		self.action = Input(shape=[self.action_dim],name='action_input')

		self.file_name = file_name #'critic_network.h5'
		self.target_file_name = target_file_name #'critic_target_network.h5'

		self.load_network()

		self.target_state = self.state
		self.target_action = self.action

		kb.set_session(tf_session)		

		self.action_gradients = tf.gradients(self.network.output,self.action)

		self.tf_session.run(tf.initialize_all_variables())


	def train(self,states,actions) :

		gradients =  self.tf_session.run(self.action_gradients, feed_dict={self.state: states, self.action: actions})[0]
		return gradients

	
	def train_target_network(self) :

		network_weights = self.network.get_weights()
		target_weights = self.target_network.get_weights()

		for i in xrange(len(network_weights)) :
			target_weights[i] = self.tau*network_weights[i] + (1-self.tau)*target_weights[i]

		self.target_network.set_weights(target_weights)



	def create_critic_network(self) :


		hidden_layer_1 =  Dense(self.hidden_layer_neurons[0], activation='relu')(self.state)

		action_layer_1 =  Dense(self.hidden_layer_neurons[1], activation='linear')(self.action)
		intermediate_layer = Dense(self.hidden_layer_neurons[1], activation='linear')(hidden_layer_1)
		merged_layer = merge([intermediate_layer,action_layer_1],mode='sum')
		hidden_layer_2 = Dense(self.hidden_layer_neurons[1],activation='relu')(merged_layer)

		output_layer = Dense(self.action_dim,activation='linear')(hidden_layer_2)

		network = Model(input=[self.state,self.action],output=output_layer)
		
		adam = Adam(lr=self.learning_rate)
		#rms = RMSprop()

		network.compile(loss='mse',optimizer=adam)
	
		return network

	def load_network(self) :

		#Load the neural networks param from the file if it exists

		self.network = self.create_critic_network()
		self.target_network = self.create_critic_network()

		try :
			self.network.load_weights(self.file_name)	

			self.target_network.load_weights(self.target_file_name)

			print self.file_name + ' loaded'
			print self.target_file_name + ' loaded'

		except IOError:
			print 'Initialized Critic Networks at ' + self.file_name + ' and ' + self.target_file_name
			
		self.weights = self.network.trainable_weights
		self.target_weights = self.target_network.trainable_weights


	def save_network(self) :

		self.network.save_weights(self.file_name)
		self.target_network.save_weights(self.target_file_name)

		print self.file_name + ' saved'
		print self.target_file_name + ' saved'	




#Create a framework for finding the optimal control policy using Deep Deterministic Policy Gradient Algorithm
class DDPG:

    #Initialize the QLearner
    def __init__(self,  drone = 'ardrone' , param = 'Thrust', controller = 'AI', setpoint = 0.4, state_dim = 2, action_dim = 1,learning_rate = 0.01, learning_rate_actor = 0.0001, learning_rate_critic = 0.000, discount_factor = 0.95, epsilon = 1.0, buffer_size = 100000, mini_batch_size = 40, tau = 0.001, hidden_layer_neurons_actor=[100,200], hidden_layer_neurons_critic=[100,200]) :

	self.drone = drone
	self.learning_rate_actor = learning_rate_actor
	self.learning_rate_critic = learning_rate_critic
	self.discount_factor = discount_factor
	self.setpoint = setpoint
	self.epsilon = epsilon
	self.replay_buffer = []
	self.buffer_size = buffer_size
	self.mini_batch_size = mini_batch_size
	self.initialized = False
	self.state = (0.0,0.0)
	self.initial_state = (0.0,0.0)
	self.current_state = (0.0,0.0)
	self.cmd_vel = 0.0
	self.epochs = 0
	self.current_action = 0.0 
	self.episode = 1
	self.count = 0.0
	self.total_reward = 0.0
	self.param = param
	self.controller = controller
	self.min_value = -1.0
	self.max_value = 1.0
	self.control_factor = 3.0
	self.learning_rate = learning_rate

		
	self.state_dim = state_dim
	self.action_dim = action_dim

	tfconfig = tf.ConfigProto()
	tfconfig.gpu_options.allow_growth = True
	tf_session = tf.Session(config=tfconfig)
	numpy.random.seed(1337)
	kb.set_session(tf_session)

	self.actor_file_name = '../models/' + drone + '_' + param + '_' + str(self.min_value) + '_' + str(self.max_value) + '_' + controller +'_actor_network.h5'

	self.actor_target_file_name = '../models/' + drone + '_' + param + '_' + str(self.min_value) + '_' + str(self.max_value) + '_' + controller +'_target_actor_network.h5'

	self.critic_file_name = '../models/' + drone + '_' + param + '_' + str(self.min_value) + '_' + str(self.max_value) + '_' + controller +'_critic_network.h5'

	self.critic_target_file_name = '../models/' + drone + '_' + param + '_' + str(self.min_value) + '_' + str(self.max_value) + '_' + controller +'_target_critic_network.h5'

	self.actor = Actor(tf_session, state_dim, action_dim, self.actor_file_name, self.actor_target_file_name, learning_rate = learning_rate_actor,tau = tau, hidden_layer_neurons = hidden_layer_neurons_actor)

	self.critic = Critic(tf_session, state_dim, action_dim, self.critic_file_name, self.critic_target_file_name, learning_rate = learning_rate_critic,tau = tau, hidden_layer_neurons = hidden_layer_neurons_critic)


	if self.controller == 'PID' :
		
		if self.param == 'Thrust':
			self.kp = 5.0
			self.kd = 81.5

		elif self.param == 'Pitch':
			self.kp = 1
			self.kd = 1

		elif self.param == 'Roll':
			self.kp = 1
			self.kd = 1

	self.replay_count = 0

	cmd_topic = '/'+ self.drone+'/command/roll_pitch_yawrate_thrust'
	sub_topic = '/'+ self.drone+'/ground_truth/position'
	self.cmd_publisher = rospy.Publisher(cmd_topic, RollPitchYawrateThrust, queue_size = 1)
	self.state_publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 1)
	rospy.Subscriber(sub_topic, PointStamped, self.get_state)
	
	time.sleep(0.2)

    #Take an action
    def run(self) :

	current_state = self.state
	
	current_action = self.get_action(current_state)		

	#Execute the current action
	self.execute_action(current_action[0][0])

	self.current_state = current_state
	self.current_action = current_action[0]
	self.count += 1.0
	
	print "state = "+ str(current_state) + ", action = " + str(round(current_action,2))


    def execute_action(self,action) :

	cmd = RollPitchYawrateThrust()
	cmd.header.stamp = rospy.Time.now()
	self.cmd_vel = action*self.control_factor
	cmd.thrust.z = 15.0 + self.cmd_vel#2.0*self.current_state[0] +12*self.current_state[1]#
	
	print "Thrust = " + str(self.cmd_vel) #str(cmd.thrust.z)#

	self.cmd_publisher.publish(cmd)	
	

    #Update the Q(s,a) table	
    def update(self) :

	#Find the next state after the second player has played
	next_state = self.state

	print "next_state = " + str(next_state)

	#Store the information in a batch and randomly sample a mini batch from this and keep replaying it 
	#While updating both the policy and mini batch
	reward = self.get_reward(next_state)
	print 'reward = ' + str(reward)
	self.experience_replay(self.current_state, self.current_action, reward , next_state)

	self.epochs += 1

	#Save policy once in every 10000 episodes
	if self.epochs % 1000 == 0 :
		#Save the updated policy
		self.actor.save_network()
		self.critic.save_network()
	
	return reward	
	


    def experience_replay(self, state, action, reward, new_state) :
	
	#If the buffer is still not filled up, keep adding information to the buffer
	if len(self.replay_buffer) < self.buffer_size :

		self.replay_buffer.append((state, action, reward, new_state))

	else:
		#After the buffer is filled, keep replacing old information with new information
		if self.replay_count < self.buffer_size - 1 :
			self.replay_count += 1
		
		else :
			self.replay_count = 0

		self.replay_buffer[self.replay_count] = (state, action, reward, new_state)
	

	#Randomly sample a mini batch from the replay buffer

	if len(self.replay_buffer) < self.mini_batch_size :

		mini_batch = random.sample(self.replay_buffer,len(self.replay_buffer))

	else :

		mini_batch = random.sample(self.replay_buffer,self.mini_batch_size)

	training_states = []

	training_actions = []

	training_outputs = []

	reward = 0.0

	#For all the tuples in the mini batch , update the current value based on the reward and append the target value to training output 
	#and append the current value to the target input
	for memory in mini_batch :

		current_state, current_action, immediate_reward, next_state = memory

		#Use the target Q network for getting the next value
		target_q_value = self.get_value(next_state)
	
		updated_value = 0.0

		reward += immediate_reward

		if immediate_reward == 1.0 or immediate_reward == -1.0 :

			updated_value = immediate_reward + 0.0*target_q_value

		else :

			updated_value = immediate_reward + 0.0*self.discount_factor*target_q_value

		updated_value = immediate_reward + 0.0*target_q_value

		training_states.append(current_state)

		training_actions.append(current_action)

		training_outputs.append(updated_value.reshape(len(target_q_value),))

	self.total_reward += reward /len(mini_batch)

	training_states = numpy.array(training_states)

	training_actions = numpy.array(training_actions)

	training_outputs = numpy.array(training_outputs)
	
	#Train the function approximator to fit the mini_batch

	self.train_batch(training_states, training_actions, training_outputs)



    def train_batch(self, training_states, training_actions, training_outputs) :

	loss = self.critic.network.train_on_batch([training_states, training_actions], training_outputs)
	updated_actions = self.actor.network.predict(training_states)
	critic_gradients = self.critic.train(training_states,updated_actions)
	self.actor.train(training_states, critic_gradients)
	self.actor.train_target_network()
	self.critic.train_target_network()

	print "Loss = " + str(loss)

    #Find the best action for a particular state
    def get_action(self, state):
			
	action = numpy.zeros([1,self.action_dim])
	action[0][0] = self.actor.network.predict(state.reshape(1,state.shape[0]))[0][0]

	if random.random() > self.epsilon :

		print "Predicted Action = " + str(action[0][0])

	else :
		if self.controller == 'PID' :
			PID = ((self.kp * self.current_state[0]) + (self.kd* self.current_state[1]))/self.control_factor
			action[0][0] +=  max(min(PID,self.max_value) , self.min_value) + numpy.random.uniform(-0.07,0.07)
			print " PID Action = " + str(action) 

		else:
			noise = max(self.epsilon,0)*(self.ornstein_uhlenbeck_randomizer(self.current_action, 0.6 , 0.0, 0.3) )# + random.uniform(-0.04,0.0))
			action[0][0] += noise
			print 'randomized action = '+ str(action[0][0]) +' ,noise = ' +str(noise)

	return action

    def ornstein_uhlenbeck_randomizer(self, x, mu, theta, sigma) :

	return theta*(mu-x) + sigma*numpy.random.randn(1)


    def get_value(self,state) :

	q_value = self.critic.target_network.predict([state.reshape(1,state.shape[0]),self.actor.target_network.predict(state.reshape(1,state.shape[0]))])

	return q_value


    #Reward Function
    def get_reward(self,state) :

	#reward = - (3*abs(state[0])) - (10*abs(state[1]))

	reward = math.exp(-abs(state[0])/(0.1*self.setpoint*3))*math.exp(-abs(state[1])/(0.1*self.setpoint*3))

	#print "reward = " + str(reward)

	#reward = 0.0

#	if abs(state[0]) >= abs(self.initial_state[0]) :
#		reward = -1.0

#	elif abs(state[0]) <= 0.05 and abs(state[1]) <=0.001 :
#		reward = 1.0

#	if abs(state[0]) <= 0.05 and abs(state[1]) <=0.001 :
#		reward = 1.0

	return reward

    #Get state of the game
    def get_state(self, msg) :
	
	state = round(self.setpoint - msg.point.z, 5)
	self.state = numpy.hstack((state, round(state - self.current_state[0],5)))

	if self.initialized is False :
		self.initial_state = numpy.hstack((round(state,2), 0.0))
		self.state = self.initial_state
		self.initialized = True

  

    def decrement_epsilon(self,value) :

	if self.epsilon > 0.01 :
		self.epsilon -= 1.0/value

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
	self.current_state = self.initial_state
	self.initialized = False	
	self.reset_flag = False

	time.sleep(0.2)
	
	print ' \n \n \n resetting \n \n \n'




	
	
