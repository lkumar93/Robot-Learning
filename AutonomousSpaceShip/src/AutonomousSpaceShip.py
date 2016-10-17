#
# THIS IS AN IMPLEMENTATION OF A SELF DRIVING SPACE SHIP
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

import SimpleGUICS2Pygame.simpleguics2pygame as simplegui
import math
import random

###########################################
##
##	VARIABLES
##
###########################################

WIDTH = 800
HEIGHT = 600
score = 0
lives = 3
time = 0
started = False
previous_closest_obstacle = None
previous_angle = 0.0
Manual = False
Automatic = False
max_number_of_rocks = 4

###########################################
##
##	CLASSES
##
###########################################

#Create a class for holding information of images
class ImageInfo:
    def __init__(self, center, size, radius = 0, lifespan = None, animated = False):
        self.center = center
        self.size = size
        self.radius = radius
        if lifespan:
            self.lifespan = lifespan
        else:
            self.lifespan = float('inf')
        self.animated = animated

    def get_center(self):
        return self.center

    def get_size(self):
        return self.size

    def get_radius(self):
        return self.radius

    def get_lifespan(self):
        return self.lifespan

    def get_animated(self):
        return self.animated


#Create a class for the Space Ship
class Ship:

    def __init__(self, pos, vel, angle, image, info):
        self.pos = [pos[0], pos[1]]
        self.vel = [vel[0], vel[1]]
        self.thrust = False
        self.angle = angle
        self.angle_vel = 0
        self.image = image
        self.image_center = info.get_center()
        self.image_size = info.get_size()
        self.radius = info.get_radius()
	self.closest_obstacle= None
	self.distance_to_closest_obstacle = 1000
	self.angle_to_closest_obstacle = 0
        
    def draw(self,canvas):
        if self.thrust:
            canvas.draw_image(self.image, [self.image_center[0] + self.image_size[0], self.image_center[1]] , self.image_size,
                              self.pos, self.image_size, self.angle)
        else:
            canvas.draw_image(self.image, self.image_center, self.image_size,
                              self.pos, self.image_size, self.angle)
       
    def update(self):        
	
        self.angle += self.angle_vel
        
        if(abs(math.degrees(self.angle)) > 360) :
		 self.angle = 0
	
        self.pos[0] = (self.pos[0] + self.vel[0]) % WIDTH
        self.pos[1] = (self.pos[1] + self.vel[1]) % HEIGHT

       
        if self.thrust:
            acc = angle_to_vector(self.angle)
            self.vel[0] += acc[0] * .3
            self.vel[1] += acc[1] * .3
            self.vel[0] *= .99
            self.vel[1] *= .99
	else :
	    self.vel[0] *= 0
            self.vel[1] *= 0
            
    def set_thrust(self, on):
        self.thrust = on
        if on:
            ship_thrust_sound.rewind()
            ship_thrust_sound.play()
        else:
            ship_thrust_sound.pause()
       
    def increment_angle_vel(self, magnitude = None):
	if magnitude is None :
        	self.angle_vel += .1
	else :
		self.angle_vel += magnitude
        
    def decrement_angle_vel(self, magnitude = None):
	if magnitude is None:
        	self.angle_vel -= .1
	else :
		self.angle_vel -= magnitude

    def stop_rotating(self):
	self.angle_vel = 0
        
    def shoot(self):
        global missile_group
        forward = angle_to_vector(self.angle)
        missile_pos = [self.pos[0] + self.radius * forward[0], self.pos[1] + self.radius * forward[1]]
        missile_vel = [self.vel[0] + 6 * forward[0], self.vel[1] + 6 * forward[1]]
        a_missile = Sprite(missile_pos, missile_vel, self.angle, 0, missile_image, missile_info, missile_sound)
           
        missile_group.add(a_missile)
        
    def obstacle_detection(self, ObstacleGroup) :

	distance_to_closest_obstacle = 10000
	angle_to_closest_obstacle = 0
	closest_obstacle = None
	
	for g in list(ObstacleGroup) :
		obstacle_position = g.get_position()
		obstacle_radius = g.get_radius()
		horizontal_distance = (self.pos[0] - obstacle_position[0])
		vertical_distance = (self.pos[1] - obstacle_position[1])
		offset = self.radius + obstacle_radius
		euclidean_distance = dist(self.pos, obstacle_position) - offset
		angle = math.degrees(math.atan2(vertical_distance,horizontal_distance))
		
		if euclidean_distance < distance_to_closest_obstacle :
			distance_to_closest_obstacle = euclidean_distance
			angle_to_closest_obstacle = angle
			closest_obstacle = g

	self.closest_obstacle = closest_obstacle
	self.distance_to_closest_obstacle = distance_to_closest_obstacle	
	angle_to_closest_obstacle = -((self.get_angle() - angle_to_closest_obstacle)%360 - 180)
	self.angle_to_closest_obstacle = angle_to_closest_obstacle 

	return self.closest_obstacle

    def get_position(self):
        return self.pos
    
    def get_radius(self):
        return self.radius

    def get_angle(self):
	if math.degrees(self.angle) < -180 :
		return math.degrees(self.angle) + 360.0
	if math.degrees(self.angle) > 180 :
		return math.degrees(self.angle) - 360.0
	else :
		return math.degrees(self.angle)

    def get_distance_to_closest_obstacle(self) :
	return self.distance_to_closest_obstacle

    def get_angle_to_closest_obstacle(self) :
	return self.angle_to_closest_obstacle



	
#Create a class to represent objects other the space ship in the environment    
class Sprite:
    def __init__(self, pos, vel, ang, ang_vel, image, info, sound = None):
        self.pos = [pos[0],pos[1]]
        self.vel = [vel[0],vel[1]]
        self.angle = ang
        self.angle_vel = ang_vel
        self.image = image
        self.image_center = info.get_center()
        self.image_size = info.get_size()
        self.radius = info.get_radius()
        self.lifespan = info.get_lifespan()
        self.animated = info.get_animated()
        self.age = 0
        if sound:
            sound.rewind()
            sound.play()
   
    def draw(self, canvas):
              
        if self.animated :
       
            canvas.draw_image(explosion_image, [ explosion_info.get_center()[0] + self.age * explosion_info.get_size()[0], explosion_info.get_center()[1]  ], explosion_info.get_size(),
                          self.pos, self.image_size, self.angle)
        
            
        else :    
        
            canvas.draw_image(self.image, self.image_center, self.image_size,
                          self.pos, self.image_size, self.angle)

    def update(self):
        # update angle
        self.angle += self.angle_vel
        
        # update position
        self.pos[0] = (self.pos[0] + self.vel[0]) % WIDTH
        self.pos[1] = (self.pos[1] + self.vel[1]) % HEIGHT
        
        self.age += 1
        
        if self.age >= self.lifespan :
            return True
        else:
            return False
        
    def get_position(self):
        return self.pos
    
    def get_radius(self):
        return self.radius    
        
    def collide(self , ship ) :
        
        ship_pos = ship.get_position()
        ship_radius = ship.get_radius()
        
        horizontal_distance = abs(self.pos[0] - ship_pos[0])
        vertical_distance = abs(self.pos[1] - ship_pos[1])
        
        if ( horizontal_distance < (self.radius + ship_radius) and vertical_distance < (self.radius + ship_radius) ) :
            return True
        else :
            return False

    def set_angle_vel(self,ang_vel) :
	self.angle_vel = ang_vel





###########################################
##
##	FUNCTIONS
##
###########################################

#Combinational Logic For Playing The Game Automatically
def auto_play():
    global started, missile_group,rock_group, previous_angle

    #Define PD Controller Constants
    kp = 0.0000115
    kd = 0.00005

    #If there are any rocks in the environment execute the logic or else freeze the ship
    if len(rock_group) is not 0:

	#If the nearest obstacle is in shootable angle, else stop rotation of ship
	if abs(my_ship.get_angle_to_closest_obstacle()) > 2 :
		
		#Implement Proportional-Derivative (PD) Controller
		Error = my_ship.get_angle_to_closest_obstacle()
		dError = previous_angle - my_ship.get_angle_to_closest_obstacle()
		magnitude = kp*Error + kd*dError
		my_ship.increment_angle_vel(magnitude)
		previous_angle = my_ship.get_angle_to_closest_obstacle()
	
		#To prevent jerky thrusting define a threshold outside of the shootable angle to thrust forward 
		if my_ship.get_distance_to_closest_obstacle() > 60 and abs(my_ship.get_angle_to_closest_obstacle()) < 3 : 
			my_ship.set_thrust(True)
		else :
			my_ship.set_thrust(False)
	else :
		previous_angle = 0.0
		my_ship.stop_rotating() 		

		#If nearest obstacle is in shootable range then fire missile or else thrust forward
		if my_ship.get_distance_to_closest_obstacle() < 60 :
			my_ship.set_thrust(False)
			if len(missile_group) is 0 :
				my_ship.shoot()
			
		else :
			my_ship.set_thrust(True)
	
    else :
	my_ship.set_thrust(False)
	my_ship.stop_rotating() 
		
#Key pressed logic          
def keydown(key):
    global Manual
    if Manual:
	    if key == simplegui.KEY_MAP['left']:
		my_ship.decrement_angle_vel()
	    elif key == simplegui.KEY_MAP['right']:
		my_ship.increment_angle_vel()
	    elif key == simplegui.KEY_MAP['up']:
		my_ship.set_thrust(True)
	    elif key == simplegui.KEY_MAP['space']:
		my_ship.shoot()

#key released logic        
def keyup(key):
    global Manual
    if Manual:
	    if key == simplegui.KEY_MAP['left']:
		my_ship.increment_angle_vel()
	    elif key == simplegui.KEY_MAP['right']:
		my_ship.decrement_angle_vel()
	    elif key == simplegui.KEY_MAP['up']:
		my_ship.set_thrust(False)
        

#When mouse is clicked on screen ,this function is called
def click(pos):
    global started , lives , score , explosion_group, Manual, Automatic
    lives = 3 
    score = 0
    explosion_group = set()

    size = [100, 100]

    inmanualwidth = (200 ) < pos[0] < (200 + size[0] )
    inmanualheight = (500 - size[1] ) < pos[1] < (500)

    inautomaticwidth = (500 ) < pos[0] < (500 + size[0] )
    inautomaticheight = (500 - size[1]) < pos[1] < (500)

    if (not started) :
	#Check if Manual or Automatic Option was selected
	if inmanualwidth and inmanualheight :
        	started = True
		Manual = True
		Automatic = False
	if inautomaticwidth and inautomaticheight:
		started = True
		Automatic = True
		Manual = False

#This is called approximately 60 times a second, all the updates to the frame happens here
def draw(canvas):
    global time, started,rock_group,lives , score , missile_group, explosion_group, previous_closest_obstacle, automatic
    time += 1
    wtime = (time / 4) % WIDTH
    center = debris_info.get_center()
    size = debris_info.get_size()
    canvas.draw_image(nebula_image, nebula_info.get_center(), nebula_info.get_size(), [WIDTH / 2, HEIGHT / 2], [WIDTH, HEIGHT])
    canvas.draw_image(debris_image, center, size, (wtime - WIDTH / 2, HEIGHT / 2), (WIDTH, HEIGHT))
    canvas.draw_image(debris_image, center, size, (wtime + WIDTH / 2, HEIGHT / 2), (WIDTH, HEIGHT))

    my_ship.draw(canvas)

    my_ship.update()
    
    
    if not started:

        canvas.draw_image(splashoverlay_image, splashoverlay_info.get_center(), 
                          splashoverlay_info.get_size(), [WIDTH / 2, HEIGHT / 2], 
                          splashoverlay_info.get_size())

        canvas.draw_text("Lakshman Kumar's", [260, 150], 40, "Green")
	canvas.draw_text("Manual", [200, 500], 40, "Green")
	canvas.draw_text("Automatic", [500, 500], 40, "Green")
	canvas.draw_text("Click any option", [350, 460], 20, "Green")
        
    else:   

        process_sprite_group(rock_group , canvas)
        process_sprite_group(missile_group , canvas)
        process_sprite_group(explosion_group , canvas)

	closest_obstacle = my_ship.obstacle_detection(rock_group)

	if closest_obstacle is not None :
		closest_obstacle.set_angle_vel(1)   
	
	if previous_closest_obstacle is not None and previous_closest_obstacle is not closest_obstacle :
		previous_closest_obstacle.set_angle_vel(0)

	previous_closest_obstacle = closest_obstacle
        
        if group_collide(rock_group,my_ship):
             lives -=1
        group_group_collide(rock_group,missile_group)
        soundtrack.play()
        
        if lives <= 0 :
            started = False
            lives = 0
            rock_group = set()
            soundtrack.rewind()

	if Automatic: 
		auto_play()
  
    if started :
	    canvas.draw_text("Lives", [50, 50], 22, "White")
	    canvas.draw_text("Score", [680, 50], 22, "White")
	    canvas.draw_text("Distance", [580, 50], 22, "White")
	    canvas.draw_text("Angle", [480, 50], 22, "White")
	    canvas.draw_text("Nearest Obstacle", [490, 25], 22, "White")
	    canvas.draw_text(str(lives), [50, 80], 22, "White")
	    canvas.draw_text(str(score), [680, 80], 22, "White")   
	    canvas.draw_text(str(math.ceil(my_ship.get_distance_to_closest_obstacle())), [580, 80], 22, "White")  
	    canvas.draw_text(str(math.ceil(my_ship.get_angle_to_closest_obstacle())), [480, 80], 22, "White")  
	    

#Spawn rocks if its less than the required threshold  
def rock_spawner():
    global  rock_group, started , my_ship, max_number_of_rocks
    rock_pos = [random.randrange(75, WIDTH-75), random.randrange(75, HEIGHT-75)]
    rock_vel = [0,0]
    rock_avel = 0
    a_rock = Sprite(rock_pos, rock_vel, 0, rock_avel, asteroid_image, asteroid_info)
    if(len(rock_group) < max_number_of_rocks and started and (not a_rock.collide(my_ship)) ) :
        rock_group.add(a_rock)
        

#Draw sprite objects on the frame
def process_sprite_group(group, canvas):    
    
    for g in list(group) :
        g.draw(canvas)
        if g.update() :
            group.remove(g)
        
#Check if a group (rocks,missiles) has collided with any other object( a space ship, a missile etc)
def group_collide(group,other_object) :
    
    global explosion_group
    for g in list(group) :
        if g.collide(other_object):
            explosion = Sprite(g.get_position(), [0,0], 0, 0, explosion_image, explosion_info,explosion_sound)
            explosion_group.add(explosion)
            
            group.remove(g)
            return True
    return False

#Check if two groups have collided (Rocks and Missiles)
def group_group_collide(group1 , group2):
    
    global score;
    for g in list(group1):
        if group_collide(group2,g) :
            group1.discard(g)
            score +=1

def angle_to_vector(ang):
    return [math.cos(ang), math.sin(ang)]

#Euclidean Distance
def dist(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

###########################################
##
##	MAIN FUNCTION
##
###########################################	

if __name__ == '__main__':

	#Create all the required image objects
    
	debris_info = ImageInfo([320, 240], [640, 480])
	debris_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/debris2_blue.png")

	nebula_info = ImageInfo([400, 300], [800, 600])
	nebula_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/nebula_blue.f2014.png")

	splashoverlay_info = ImageInfo([WIDTH/2, HEIGHT/2], [WIDTH,HEIGHT])
	splashoverlay_image = simplegui.load_image("http://sfstory.free.fr/images/Affiches/starwars800x600.jpg")

	ship_info = ImageInfo([45, 45], [90, 90], 35)
	ship_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/double_ship.png")

	missile_info = ImageInfo([5,5], [10, 10], 3, 50)
	missile_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/shot2.png")

	asteroid_info = ImageInfo([45, 45], [90, 90], 40)
	asteroid_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/asteroid_blue.png")

	explosion_info = ImageInfo([64, 64], [128, 128], 17, 24, True)
	explosion_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/explosion_alpha.png")

	#Create game sounds

	soundtrack = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/soundtrack.mp3")
	missile_sound = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/missile.mp3")
	missile_sound.set_volume(.5)
	ship_thrust_sound = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/thrust.mp3")
	explosion_sound = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/explosion.mp3")

	#Create the game screen
	frame = simplegui.create_frame("Asteroids", WIDTH, HEIGHT)

	#Create the ship
	my_ship = Ship([WIDTH / 2, HEIGHT / 2], [0, 0], 0, ship_image, ship_info)

	#Initialize sprites
	rock_group = set()
	missile_group = set()
	explosion_group = set()

	#Set appropriate helper functions for frame object
	frame.set_keyup_handler(keyup)
	frame.set_keydown_handler(keydown)
	frame.set_mouseclick_handler(click)
	frame.set_draw_handler(draw)

	#Create a timer to run rock_spawner every 10 ms
	timer = simplegui.create_timer(10.0, rock_spawner)

	#Start the timer
	timer.start()

	#Start the game screen
	frame.start()

