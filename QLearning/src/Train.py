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

import pygame, sys
import matplotlib.pyplot as plotter
import time

from pygame.locals import QUIT, MOUSEBUTTONUP
from QLearning import QLearner
from TicTacToe import Board


###########################################
##
##	VARIABLES
##
###########################################

epochs = 1000
wins = 0.0
losses = 0.0
draws = 0.0
winning_percentages_50_epoch = []
losing_percentages_50_epoch = []
drawing_percentages_50_epoch = []

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

	#Initialize the game
	TicTacToeBoard = Board(grid_size=3, box_size=100, border=50, line_width=10)
	
	#Initialize Q Learning Player
	qlearner1 = QLearner(TicTacToeBoard, 1)
	qlearner2 = QLearner(TicTacToeBoard, 2)

	#Train the qlearner upto the number of epochs required
	for i in range(1,epochs) :

		#Qlearner1 plays X
		qlearner1.play()

		#Qlearner2 plays O
		qlearner2.play()

		#Update policy of Qlearner 1
		qlearner1.update_policy()

		# While the game is not over
		while TicTacToeBoard.check_game_over() is False :

			#Train Qlearner1
			qlearner1.play()

			#Update policy of Qlearner2
			qlearner2.update_policy()

			#Update Display
			pygame.display.update()

			#Train Qlearner2
			if TicTacToeBoard.check_game_over()  is False :
				qlearner2.play()
			
			#Train Qlearner
			qlearner1.update_policy()

			#Check if user wants to quit							
			for event in pygame.event.get():
				if event.type == QUIT :
				    time.sleep(1)
				    qlearner1.save_policy()
				    qlearner2.save_policy()
				    print "Exiting Game"
				    pygame.quit()
				    sys.exit()

			#Update Display
			pygame.display.update()

			#Update clock frequency
			clock.tick(100)	

		#Update policy of qlearner2
		qlearner2.update_policy()

		#Keep track of number of total wins, losses and draws

		if TicTacToeBoard.check_for_winner() == 1:
			wins = wins + 1

		elif TicTacToeBoard.check_for_winner() == 2:
			losses = losses + 1

		else :
			draws = draws + 1

		#Calculate total number of games played
		total = wins+losses+draws
		
		#Calculate cumulative percentage of winning,losing and drawing
		percentage_wins = round((wins*100)/total,2)
		percentage_losses = round((losses*100)/total,2)
		percentage_draws = round((draws*100)/total,2)

		#Print the information
		statement = "epoch : " + str(i) + " wins : " + str(percentage_wins) + " losses : " + str(percentage_losses) + " draws : " + str(percentage_draws)
		print statement

		#Store the percentage of winning,losing and drawing every 50 epochs
		if i%50 == 0 and i > 100:
			winning_percentages_50_epoch.append(percentage_wins)
			losing_percentages_50_epoch.append(percentage_losses)
			drawing_percentages_50_epoch.append(percentage_draws)

		#Keep track of number of episodes completed
		TicTacToeBoard.increment_epochs()

		#Reset the game
		TicTacToeBoard.reset()	
	
		
	#Store the number of epochs ( x50 ) passed	
	range_of_values = range(2,len(winning_percentages_50_epoch)+2)

	#Plot the performance 

	plotter.figure()		

	plotter.plot(range_of_values,winning_percentages_50_epoch,'g',label='Wins' )
	plotter.plot(range_of_values,losing_percentages_50_epoch,'r',label='Losses' )
	plotter.plot(range_of_values,drawing_percentages_50_epoch,'b',label='Draws' )

	plotter.xlabel('X50 Number of Episodes Completed')
	plotter.ylabel('Cumulative Percentage')

	plotter.legend(loc='upper right', shadow=True)

	TitleText = " PERFORMANCE OF Q LEARNING TIC TAC TOE WITH LOOK UP TABLE \n \n" + " LEARNING RATE =  " +str(qlearner1.learning_rate) + " , DISCOUNT FACTOR =  " + str(qlearner1.discount_factor) + " , EPSILON = " + str(qlearner1.epsilon)
	plotter.title(TitleText)

	plotter.show()
	



