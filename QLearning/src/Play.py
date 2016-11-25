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


import pygame, sys, getopt
from pygame.locals import QUIT, MOUSEBUTTONUP
import time

from QLearning import QLearner
from TicTacToe import Board

###########################################
##
##	VARIABLES
##
###########################################

wins = 0.0
losses = 0.0
draws = 0.0

func_approx_flag = False
exp_replay_flag = None

###########################################
##
##	MAIN FUNCTION
##
###########################################


if __name__ == '__main__':

	#Initialize pygame gui
	pygame.init()

	#Initialize pygame clock
	clock = pygame.time.Clock()

	#Parse command line arguments to check if the user wants to enable function approximation or experience replay

	argv = sys.argv[1:]

   	try:
      		opts, args = getopt.getopt(argv,"f:e:r:",["func_approx=","exp_replay="])

   	except getopt.GetoptError:
      		print 'Usage: python Play.py -f <bool> -e <bool>'
      		sys.exit(2)
   	
	for opt, arg in opts:

     		if opt in ("-f", "--func_approx"):

			if arg == 'True' :
				func_approx_flag = True

			elif arg == 'False' :
				func_approx_flag = False

			else :
				print 'Usage: python Play.py -f <bool> -e <bool>'

				sys.exit(2)

		elif opt in ("-e", "--exp_replay") :

			if arg == 'True' :
				exp_replay_flag = True

			elif arg == 'False' :
				exp_replay_flag = False

			else :
				print 'Usage: python Play.py -f <bool>  -e <bool>'

				sys.exit(2)
		

	#Initialize the game
	TicTacToeBoard = Board(grid_size=3, box_size=100, border=50, line_width=10)

	#Initialize Q Learning Player
	qlearner = QLearner(TicTacToeBoard, q_id = 1, function_approximation = func_approx_flag,epsilon = 0.0,replay = exp_replay_flag)
			

	#Loop Forever
	while True:

		# While the game is not over
		while TicTacToeBoard.check_game_over() is False :

			# Ask the qlearner to play first
			qlearner.play()	
	
			#Initialize a flag for checking whether the user has played or not
			user_played = False
		
			#Update Display
			pygame.display.update()

			#Wait till the user has played
			while user_played is False and TicTacToeBoard.check_game_over() is False :
				
				#Check for user actions		
				for event in pygame.event.get():
					if event.type == QUIT :
					    qlearner.save_policy()
					    print "Exiting Game"
					    pygame.quit()
					    sys.exit()
					elif event.type == MOUSEBUTTONUP:
					    x, y = event.pos
					    user_played = TicTacToeBoard.process_click(x, y)
					    pygame.event.clear()
	   			    	    break

			#Update qlearner's policy
			qlearner.update_policy()	
					   
			#Update Display
			pygame.display.update()

			#Set clock frequency
			clock.tick(100)	

		#Keep track of number of total wins, losses and draws

		if TicTacToeBoard.check_for_winner() == 1:
			wins = wins + 1

		elif TicTacToeBoard.check_for_winner() == 2:
			losses = losses + 1

		else :
			draws = draws + 1

		#Calculate total number of games played
		total = wins+losses+draws

		#Print the information
		statement = "Number Of Games : " + str(total) + " Computer Score : " + str(wins) + " Your Score : " + str(losses) + " Draws : " + str(draws)
		print statement
		

		#Once the game is over sleep for a while 
		time.sleep(1)

		#Reset the game
		TicTacToeBoard.reset()
		





