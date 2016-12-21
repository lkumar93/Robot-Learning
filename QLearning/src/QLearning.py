#
# THIS IS AN IMPLEMENTATION OF Q LEARNING ALGORITHM TO PLAY TIC TAC TOE
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

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

###########################################
##
##	CLASSES
##
###########################################

#Create a framework for finding the optimal winning policy using Q Learning Algorithm
class QLearner:

    #Initialize the QLearner
    def __init__(self, board, q_id = 1, function_approximation = False, learning_rate = 0.6, discount_factor = 0.9, epsilon = 0.5, buffer_size = 80, mini_batch_size = 40, replay = None) :

	self.board = board
	self.learning_rate = learning_rate
	self.discount_factor = discount_factor
	self.epsilon = epsilon
	self.q_id = q_id
	self.function_approximation = function_approximation
	self.replay_buffer = []
	self.buffer_size = buffer_size
	self.mini_batch_size = mini_batch_size

	if replay is None :

		self.replay = function_approximation

	else :
		self.replay = replay

	self.replay_count = 0

	if function_approximation :
		self.file_name = '../policy/policy_q'+ str(self.q_id) +'_fa.h5'
	else :
		self.file_name = '../policy/policy_q'+ str(self.q_id) +'.json'

	self.load_policy()
	


    #Take an action
    def play(self) :


	#Get the current state
	current_state = self.get_state()

	#Check which moves are legal
	legal_moves = self.board.check_for_free_boxes()

	current_action = None
	current_value = 0

	if self.function_approximation :
	
		if random.random() > self.epsilon :

			current_action, current_value, current_q_values = self.get_best_action_value(current_state, legal_moves)

		else :

			current_action, current_value, current_q_values = self.get_best_action_value(current_state, legal_moves, random_action = True)		

		self.current_q_values = current_q_values


	else :

		#If current state is in Q Learner's policy 
		if current_state in self.policy :

			#Epsilon-Greedy Strategy : Exploitation vs Exploration
			#Select the best action with (1-epsilon) probability
			if random.random() > self.epsilon :
				#Find the best action and value
				current_action, current_value = self.get_best_action_value(current_state,legal_moves)

	 	#As current state is not in the policy, initialize a list for that particular state
		else :
			self.policy[current_state] = []

		#If the current state is not in the policy or if selecting a random action with epsilon probability 
		#Make sure you are selecting an action that has not been executed before , if all the actions have been 
		#executed before then do a random action

		if current_action is None :

			current_action, current_value = self.get_best_action_value(current_state,legal_moves,random_action = True)
					

	#Execute the current action
	self.board.qlearning_player(current_action)

	self.current_state = current_state
	self.current_action = current_action
	self.current_value = current_value


    #Update the Q(s,a) table	
    def update_policy(self) :

	#Find the next state after the second player has played
	next_state = self.get_state()

	#Initialize the value for next state action pair
	next_value = 0

	next_q_values = None


	#Check which moves are legal	
	legal_moves = self.board.check_for_free_boxes()


	if self.function_approximation :


		if self.replay :

			#Store the information in a batch and randomly sample a mini batch from this and keep replaying it 
			#While updating both the policy and mini batch

			self.experience_replay(self.current_state, self.current_action, self.get_reward(), next_state, legal_moves)


		else :

			#Get the q values for next state

			next_action, next_value, next_q_values = self.get_best_action_value(next_state,legal_moves)

			target = numpy.zeros( ( 1, len(self.current_q_values) ) )
		
			target[:] = self.current_q_values[:]

			reward = self.get_reward()

			#If not a terminal state then use only the next value

			if reward == 0 :

				target[0][self.current_action] = self.discount_factor*next_value

			#If terminal state then use only the reward

			else :
				target[0][self.current_action] = reward 

			#Train the function approximator to fit this data

			self.function_approximator.fit(self.current_state.reshape( 1,len(self.current_state)), target , batch_size = 1, nb_epoch = 1, verbose = 1) 



	else :
		

		#If the next state is in QLearner's policy, then
		if next_state in self.policy : 


			#Find the next action and the value of next state/action pair
			next_action, next_value = self.get_best_action_value(next_state,legal_moves)

		#As next state is not in the policy, initialize a list for that particular state
		else :
			self.policy[next_state] = []


		#Q(s,a) = Q(s,a) + Alpha * ( R' + Gamma*Q(s',a') - Q(s,a) )
		self.current_value = self.current_value + self.learning_rate*(self.get_reward()+ self.discount_factor*next_value - self.current_value)

		#Round off to four decimal points
		self.current_value = round(self.current_value,4)
	
		action_in_policy = False

		#If the action is in the policy for current state modify the value of the action
		for av in self.policy[self.current_state] :
			if av['action'] == self.current_action :
				action_in_policy = True
				av['value'] = self.current_value
	
		#If the action is not in the policy, augment the action value pair to that state
		if action_in_policy is not True :
			self.policy[self.current_state].append({'action':self.current_action,'value':self.current_value})


	#Save policy once in every 10000 episodes
	if self.board.epochs % 10000 == 0 :
		#Save the updated policy
		self.save_policy()	
	


    def experience_replay(self, state, action, reward, new_state, legal_moves) :
	
	#If the buffer is still not filled up, keep adding information to the buffer
	if(len(self.replay_buffer) < self.buffer_size) :

		self.replay_buffer.append((state, action, reward, new_state, legal_moves))

	else:
		#After the buffer is filled, keep replacing old information with new information
		if self.replay_count < self.buffer_size - 1 :
			self.replay_count += 1
		
		else :
			self.replay_count = 0
			self.target_q_network = self.function_approximator

		self.replay_buffer[self.replay_count] = (state, action, reward, new_state, legal_moves)

		#Randomly sample a mini batch from the replay buffer

		mini_batch = random.sample(self.replay_buffer,self.mini_batch_size)

		training_input = []

		training_output = []


		#For all the tuples in the mini batch , update the current value based on the reward and append the target value to training output 
		#and append the current value to the target input
		for memory in mini_batch :

			current_state, current_action, immediate_reward, next_state, possible_actions = memory

			current_q_values = self.function_approximator.predict(current_state.reshape(1,len(state)),batch_size = 1)

			#Use the frozen target Q network for getting the next value
			next_action, next_value, next_q_values = self.get_best_action_value(next_state, possible_actions, use_target = True)

			target = numpy.zeros((1,len(current_q_values[0])))

			target[:] = current_q_values[:]

			updated_value = 0

			if reward == 0 :
				
				updated_value = self.discount_factor*next_value

			else :
				updated_value = immediate_reward

			target[0][current_action] = updated_value

			training_input.append(current_state.reshape(len(state),))

			training_output.append(target.reshape(len(current_q_values[0]),))


		training_input = numpy.array(training_input)
	
		training_output = numpy.array(training_output)

		#Train the function approximator to fit the mini_batch

		self.function_approximator.fit(training_input, training_output, batch_size = self.mini_batch_size, nb_epoch = 10, verbose = 0)


    #Find the best action for a particular state
    def get_best_action_value(self, state, possible_actions, random_action = False, use_target = False):

	action = None
	value = 0
	q_values = None

	if self.function_approximation :

		#Decide whether to use the frozen target q network or the updated function approximator
		if use_target :
			q_values = self.target_q_network.predict(state.reshape(1,len(state)),batch_size = 1)

		else :
			q_values = self.function_approximator.predict(state.reshape(1,len(state)),batch_size = 1)

		if random_action :
			
			action = random.choice(possible_actions)
			value = q_values[0][action]

		else :
		
			#Find a legal action that has the highest value
			sorted_actions = numpy.argsort(q_values[0])[::-1]
			sorted_values = numpy.sort(q_values[0])[::-1]

			for selected_action in sorted_actions :
				if selected_action in possible_actions : 
					action = selected_action
					value = q_values[0][action]
					break

		return action, value, q_values

	else :
	
		if random_action :

			action = random.choice(possible_actions)

			for move in possible_actions:

				action_in_policy = False

				for av in self.policy[state] :

					if av['action'] == move :
						
						action_in_policy = True
						
					if av['action'] == action :

						value = av['value']

		
				if action_in_policy is False :
						
					action = move
					value = 0.0
				
					break


		else :
			sorted_av_table = []

			#Sort the actions according to the values
			sorted_av_table = sorted(self.policy[state], reverse=True, key = lambda av : av['value'])

			#Find the best action value pair as long as its a legal move and is greater than 0
			for i in range(0,len(sorted_av_table)) :
				if sorted_av_table[i]['action'] in possible_actions and sorted_av_table[i]['value'] > 0.0 :
					action = sorted_av_table[i]['action']
					value = sorted_av_table[i]['value']
					break

		return action, value


    #Initialize Neural Network
    def initialize_neural_network(self) :

	#Create a sequential neural network with 2 hidden layers of 27 and 18 neurons respectively, the output layer
	#consists of 9 neurons which maps to box numbers in the Tic Tac Toe Board

	model = Sequential()

	board_size = self.board.grid_size**2

	model.add(Dense(3*board_size, init='lecun_uniform', input_shape=(2*board_size,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))

	model.add(Dense(18*board_size, init='lecun_uniform'))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	
	model.add(Dense(board_size, init='lecun_uniform'))
	model.add(Activation('linear'))

	rms = RMSprop()
	model.compile(loss='mse', optimizer=rms)

	return model


    #Reward Function
    def get_reward(self) :

	
	if self.board.check_game_over() is True :

		#Reward of 1 or -1 based on which QLearning Player wins or loses (X or O)
		if self.board.check_for_winner() == 1 :

			if self.q_id == 1 :
				reward = 1
			else :
				reward = -1
			
		elif self.board.check_for_winner() == 2 :
			if self.q_id == 1:
				reward = -1
			else :
				reward = 1
			
		#Reward of 0.1 if the game is a draw
		else :
			reward = 0.1

	#No reward if the game is progress
	else :
		reward = 0


	return reward

    #Get state of the game
    def get_state(self) :
	
	if self.function_approximation :
		
		state = []

		my_state = []
	
		opponent_state = []

		#Binary encoding of states of this q learner and the opponent

		for box in self.board.boxes :

			#Empty Boxes
			
			if box.state == 3 :

				my_state.append(0)
				opponent_state.append(0)

			# X's
			elif box.state == 1 :

				if self.q_id == 1 :
					my_state.append(1)
					opponent_state.append(0)

				else :
					my_state.append(0)
					opponent_state.append(1)
			# O's
			else :

				if self.q_id == 1 :
					my_state.append(0)
					opponent_state.append(1)

				else :
					my_state.append(1)
					opponent_state.append(0)
				


		state.extend(my_state)
		state.extend(opponent_state)

		return numpy.array(state)

	else :
		state = 0

		#Encode the state as a 9 digit number
		for box in self.board.boxes :
			state = state*10 + box.state

		return str(state)


    def decrement_epsilon(self,value) :

	if self.epsilon > 0.1 :
		self.epsilon -= 1.0/value

    #Load the policy 
    def load_policy(self) :

	if self.function_approximation :
		#Load the neural networks param from the file if it exists
		try :
			self.function_approximator = load_model(self.file_name)
			self.target_q_network = self.function_approximator
		
		#Else initialize the neural network			
		except IOError:
		
			self.function_approximator = self.initialize_neural_network()
			self.target_q_network = self.initialize_neural_network()
			print 'Initialized Neural Network at ' + self.file_name 
			return				
		
		print self.file_name + ' loaded'


	else :

		self.policy = {}

		#Open the json file if available
		try:
			policy_file = open(self.file_name, 'r')
		except IOError:
			return

		#Load the contents of the file to QLearner's policy
		self.policy = json.load(policy_file)

		#Close the policy
		policy_file.close()

		print self.file_name + ' loaded'

    #Save the policy
    def save_policy(self) :

	if self.function_approximation :

		#Save the architecture and weights of neural network as a HDF5 file
		self.function_approximator.save(self.file_name)

	else :
		#Save the policy as a json file
		policy_file = open(self.file_name,'w')
	
		#Dump the dictionary in QLearner's file as a JSON string to policy file
		json.dump(self.policy,policy_file,sort_keys = True,indent = 4, separators=(',',': '))

		#Close the policy file
		policy_file.close()

	print self.file_name + ' saved'	
