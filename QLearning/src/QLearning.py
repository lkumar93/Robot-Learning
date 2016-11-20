#
# THIS IS AN IMPLEMENTATION OF Q LEARNING ALGORITHM ON
# TIC TAC TOE
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

###########################################
##
##	CLASSES
##
###########################################

#Create a framework for finding the optimal winning policy using Q Learning Algorithm
class QLearner:

    #Initialize the QLearner
    def __init__(self, board, q_id = 1, learning_rate = 0.6, discount_factor = 0.9, epsilon = 0.4) :

	self.board = board
	self.learning_rate = learning_rate
	self.discount_factor = discount_factor
	self.epsilon = epsilon
	self.q_id = q_id
	self.current_state = 333333333
	self.current_action = 5
	self.current_value = 0
	self.load_policy()

    #Take an action
    def play(self) :

	#Get the current state
	current_state = self.get_state()

	#Check which moves are legal
	legal_moves = self.board.check_for_free_boxes()
	
	current_action = None
	current_value = 0

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
	#Make sure if the particular action is in policy, you dont select it if it has negative values

	if current_action is None :

		loop_count = 0
		positive_value = False
		action_in_policy = False

		while loop_count < 5 and not positive_value:
			
			current_action = random.choice(legal_moves)
			
			for av in self.policy[current_state] :

				if av['action'] == current_action :
					
					action_in_policy = True
				
					if av['value'] >= 0 :

						positive_value = True			
						break

			if not action_in_policy :
				break
	
			loop_count = loop_count + 1
					

	#Execute the current action
	self.board.qlearning_player(current_action)

	self.current_state = current_state
	self.current_action = current_action
	self.current_value = current_value

    #Find the best action for a particular state
    def get_best_action_value(self, state, possible_actions):

	action = None
	value = 0
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

    #Update the Q(s,a) table	
    def update_policy(self) :

	#Find the next state after the second player has played
	next_state = self.get_state()
	
	#Initialize the value for next state action pair
	next_value = 0

	#If the next state is in QLearner's policy, then
	if next_state in self.policy : 

		#Check which moves are legal	
		legal_moves = self.board.check_for_free_boxes()

		#Find the next action and the value of next state/action pair
		next_action, next_value = self.get_best_action_value(next_state,legal_moves)

	#As next state is not in the policy, initialize a list for that particular state
	else :
		self.policy[next_state] = []


	#Q(s,a) = Q(s,a) + Alpha * ( R' + Gamma*Q(s',a') - Q(s,a) )
	self.current_value = self.current_value + self.learning_rate*(self.get_reward()+ self.discount_factor*next_value - self.current_value)

	#Round off to two decimal points
	self.current_value = round(self.current_value,2)
	
	action_in_policy = False

	#If the action is in the policy for current state modify the value of the action
	for av in self.policy[self.current_state] :
		if av['action'] == self.current_action :
			action_in_policy = True
			av['value'] = self.current_value
	
	#If the action is not in the policy, augment the action value pair to that state
	if action_in_policy is not True :
		self.policy[self.current_state].append({'action':self.current_action,'value':self.current_value})


	#Save policy once in every 100 games
	if self.board.epochs % 100 == 0 :
		#Save the updated policy
		self.save_policy()	
	
    #Load the policy 
    def load_policy(self) :

	self.policy = {}

	file_name = 'policy_q'+ str(self.q_id) +'.json'

	#Open the json file if available
	try:
		policy_file = open(file_name, 'r')
	except IOError:
		return

	#Load the contents of the file to QLearner's policy
	self.policy = json.load(policy_file)

	#Close the policy
	policy_file.close()

	print file_name + ' loaded'

    #Save the policy
    def save_policy(self) :

	#Save the policy as a json file

	file_name = 'policy_q'+ str(self.q_id) +'.json'
	policy_file = open(file_name,'w')
	
	#Dump the dictionary in QLearner's file as a JSON string to policy file
	json.dump(self.policy,policy_file)

	#Close the policy file
	policy_file.close()


    #Reward Function
    def get_reward(self) :

	
	if self.board.check_game_over() is True :

		#Reward of 1 if QLearning Player wins
		if self.board.check_for_winner() == 1 :
			if self.q_id == 1 :
				reward = 1
			else :
				reward = -1
			
		#Reward of -1 if the opponent wins
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
	
	state = 0

	#Encode the state as a 9 digit number
	for box in self.board.boxes :
		state = state*10 + box.state

	return state
		

	






	
