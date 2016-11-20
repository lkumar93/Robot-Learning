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
import itertools
import random
import time
import matplotlib.pyplot as plotter

from pygame.locals import QUIT, MOUSEBUTTONUP
from QLearning import QLearner


###########################################
##
##	VARIABLES
##
###########################################

#Color Variables
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

###########################################
##
##	CLASSES
##
###########################################

#Create a framework for creating a box in the board
class Box(object):

    #Default state is 3 which denotes empty box	
    state = 3
    
    #Initialize the box
    def __init__(self, x, y, size, board):
        self.size = size
        self.line_width = int(self.size / 40) if self.size > 40 else 1
        self.radius = (self.size / 2) - (self.size / 8)
        self.rect = pygame.Rect(x, y, size, size)
        self.board = board
    
    #Display X on the box
    def mark_x(self):
        pygame.draw.line(self.board.surface, RED, (self.rect.centerx - self.radius, self.rect.centery - self.radius), (self.rect.centerx + self.radius, self.rect.centery + self.radius), self.line_width)
        pygame.draw.line(self.board.surface, RED, (self.rect.centerx - self.radius, self.rect.centery + self.radius), (self.rect.centerx + self.radius, self.rect.centery - self.radius), self.line_width)
    
    #Display O on the box
    def mark_o(self):
        pygame.draw.circle(self.board.surface, BLUE, (self.rect.centerx, self.rect.centery), self.radius, self.line_width)

#Create a framework for creating a TicTacToe Board
class Board(object):

    #X Goes First
    turn = 1
    
    #Initialize the game
    def __init__(self, grid_size=3, box_size=200, border=20, line_width=5):
        self.grid_size = grid_size
        self.box_size = box_size
        self.border = border
        self.line_width = line_width
        surface_size = (self.grid_size * self.box_size) + (self.border * 2) + (self.line_width * (self.grid_size - 1))
        self.surface = pygame.display.set_mode((surface_size, surface_size), 0, 32)
        self.game_over = False
	self.epochs = 1
        self.setup()
    
    #Setup the board     
    def setup(self):
        pygame.display.set_caption('Tic Tac Toe')
        self.surface.fill(WHITE)
        self.draw_lines()
        self.initialize_boxes()
        self.create_winning_combinations()

    #Reset the board
    def reset(self):
	self.game_over = False
	self.turn = 1
	self.setup()
    
    #Draw the grid lines on the board
    def draw_lines(self):
        for i in xrange(1, self.grid_size):
            start_position = ((self.box_size * i) + (self.line_width * (i - 1))) + self.border
            width = self.surface.get_width() - (2 * self.border)
            pygame.draw.rect(self.surface, BLACK, (start_position, self.border, self.line_width, width))
            pygame.draw.rect(self.surface, BLACK, (self.border, start_position, width, self.line_width))
    
    #Initialize each box in the board
    def initialize_boxes(self):
        self.boxes = []
        
        top_left_numbers = []
        for i in range(0, self.grid_size):
            num = ((i * self.box_size) + self.border + (i *self.line_width))
            top_left_numbers.append(num)
        
        box_coordinates = list(itertools.product(top_left_numbers, repeat=2))
        for x, y in box_coordinates:
            self.boxes.append(Box(x, y, self.box_size, self))
    
    #Get the box index at the given pixel co-ordinate
    def get_box_at_pixel(self, x, y):
        for index, box in enumerate(self.boxes):
            if box.rect.collidepoint(x, y):
		if index in self.check_for_free_boxes():
                	return box
        return None
    
    #Get user's intent and execute the action 
    def process_click(self, x, y):
        box = self.get_box_at_pixel(x, y)
        if box is not None and not self.check_game_over():
            self.play_turn(box)
	    return True
	return False

    #Get the indices of empty spots in the board	
    def check_for_free_boxes(self) :
	index = 0
	free_boxes = []
	for box in self.boxes :
		if box.state == 3 :
			free_boxes.append(index)
		index = index + 1

	return free_boxes

    #Execute the action of the qlearning player
    def qlearning_player(self, move):

	if not self.check_game_over() :

		self.play_turn(self.boxes[move])

    #Execute the action of a random player
    def random_player(self) :

	if not self.check_game_over() :
		free_boxes = self.check_for_free_boxes()
		if len(free_boxes) > 0 :
			chosen_box = random.choice(free_boxes)	
			self.play_turn(self.boxes[chosen_box])
	
    #Mark the box with corresponding symbol    
    def play_turn(self, box):
	# 3 corresponds to empty box
        if box.state != 3:
            return

	# 1 corresponds to X
        if self.turn == 1:
            box.mark_x()
            box.state = 1
            self.turn = 2

	# 2 corresponds to O
        elif self.turn == 2:
            box.mark_o()
            box.state = 2
            self.turn = 1

        return
    
    #Create Winning Combinations
    def create_winning_combinations(self):
        self.winning_combinations = []
        indices = [x for x in xrange(0, self.grid_size * self.grid_size)]
        
        # Vertical combinations
        self.winning_combinations += ([tuple(indices[i:i+self.grid_size]) for i in xrange(0, len(indices), self.grid_size)])
        
        # Horizontal combinations
        self.winning_combinations += [tuple([indices[x] for x in xrange(y, len(indices), self.grid_size)]) for y in xrange(0, self.grid_size)]
        
        # Diagonal combinations
        self.winning_combinations.append(tuple(x for x in xrange(0, len(indices), self.grid_size + 1)))
        self.winning_combinations.append(tuple(x for x in xrange(self.grid_size - 1, len(indices)-1, self.grid_size - 1)))
    
   
    #Check which player won
    def check_for_winner(self):

	#0 corresponds to draw
        winner = 0

	#Check winner for all the combinations
        for combination in self.winning_combinations:
            states = []
	
            for index in combination:
                states.append(self.boxes[index].state)

	    #1 corresponds to X
            if all(x == 1 for x in states):
                winner = 1
	
	    #2 corresponds to O
            if all(x == 2 for x in states):
                winner = 2

        return winner
    
    #Check if the game is over	
    def check_game_over(self):

	#Check for winner
        winner = self.check_for_winner()
	
	#If the winner is X or Y
        if winner:
            self.game_over = True

	#If its a draw
        elif all(box.state in [1, 2] for box in self.boxes):
            self.game_over = True

	#If the game is over display the message
        if self.game_over:
            self.display_game_over(winner)
	    return True
	else :
	    return False
    
    #Display which player won 
    def display_game_over(self, winner):

        surface_size = self.surface.get_height()

        font = pygame.font.Font('freesansbold.ttf', surface_size / 8)

        if winner == 1:
            text = 'Player 1 won!'

	elif winner == 2:
	    text = 'Player 2 won!'
	
        else:
            text = 'Draw!'

        text = font.render(text, True, BLACK, WHITE)

        rect = text.get_rect()
        rect.center = (surface_size / 2, surface_size / 2)

        self.surface.blit(text, rect)

    def increment_epochs(self) :
	self.epochs = self.epochs + 1


