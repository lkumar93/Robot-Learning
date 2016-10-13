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

###########################################
##
##	CLASSES
##
###########################################

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
       
    def increment_angle_vel(self):
        self.angle_vel += .1
        
    def decrement_angle_vel(self):
        self.angle_vel -= .1
        
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
	angle =self.get_angle() + angle_to_closest_obstacle
	
	if angle < -180 :
		angle = angle + 270

	self.angle_to_closest_obstacle = angle 

	return self.closest_obstacle

    def get_position(self):
        return self.pos
    
    def get_radius(self):
        return self.radius

    def get_angle(self):
	return abs(math.degrees(self.angle))%360-180

    def get_distance_to_closest_obstacle(self) :
	return self.distance_to_closest_obstacle

    def get_angle_to_closest_obstacle(self) :
	return self.angle_to_closest_obstacle



	
    
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
        #print "spawning"

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
          
def keydown(key):
    if key == simplegui.KEY_MAP['left']:
        my_ship.decrement_angle_vel()
    elif key == simplegui.KEY_MAP['right']:
        my_ship.increment_angle_vel()
    elif key == simplegui.KEY_MAP['up']:
        my_ship.set_thrust(True)
    elif key == simplegui.KEY_MAP['space']:
        my_ship.shoot()
        
def keyup(key):
    if key == simplegui.KEY_MAP['left']:
        my_ship.increment_angle_vel()
    elif key == simplegui.KEY_MAP['right']:
        my_ship.decrement_angle_vel()
    elif key == simplegui.KEY_MAP['up']:
        my_ship.set_thrust(False)
        

def click(pos):
    global started , lives , score , explosion_group
    center = [WIDTH / 2, HEIGHT / 2]
    size = splash_info.get_size()
    lives = 3 
    score = 0
    explosion_group = set()
    inwidth = (center[0] - size[0] / 2) < pos[0] < (center[0] + size[0] / 2)
    inheight = (center[1] - size[1] / 2) < pos[1] < (center[1] + size[1] / 2)
    if (not started) and inwidth and inheight:
        started = True

def draw(canvas):
    global time, started,rock_group,lives , score , missile_group, explosion_group, previous_closest_obstacle
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
        
        canvas.draw_image(splash_image, splash_info.get_center(), 
                          splash_info.get_size(), [WIDTH / 2, HEIGHT / 2], 
                          splash_info.get_size())
        canvas.draw_text("Lakshman Kumar's", [250, 220], 40, "White")
        
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

  
    canvas.draw_text("Lives", [50, 50], 22, "White")
    canvas.draw_text("Score", [680, 50], 22, "White")
    canvas.draw_text("Distance", [580, 50], 22, "White")
    canvas.draw_text("Angle", [480, 50], 22, "White")
    canvas.draw_text("Ship Angle", [380, 50], 22, "White")
    canvas.draw_text(str(lives), [50, 80], 22, "White")
    canvas.draw_text(str(score), [680, 80], 22, "White")   
    canvas.draw_text(str(math.ceil(my_ship.get_distance_to_closest_obstacle())), [580, 80], 22, "White")  
    canvas.draw_text(str(math.ceil(my_ship.get_angle())+math.ceil(my_ship.get_angle_to_closest_obstacle())), [480, 80], 22, "White")  
    canvas.draw_text(str(math.ceil(my_ship.get_angle())), [380, 80], 22, "White")  

   
def rock_spawner():
    global  rock_group, started , my_ship
    rock_pos = [random.randrange(75, WIDTH-75), random.randrange(75, HEIGHT-75)]
    rock_vel = [0,0]#[random.random() * .6 - .3, random.random() * .6 - .3]
    rock_avel = 0#random.random() * .2 - .1
    a_rock = Sprite(rock_pos, rock_vel, 0, rock_avel, asteroid_image, asteroid_info)
    if(len(rock_group) < 8 and started and (not a_rock.collide(my_ship)) ) :
        rock_group.add(a_rock)
        

def process_sprite_group(group, canvas):    
    
    for g in list(group) :
        g.draw(canvas)
        if g.update() :
            group.remove(g)
        

def group_collide(group,other_object) :
    
    global explosion_group
    for g in list(group) :
        if g.collide(other_object):
            explosion = Sprite(g.get_position(), [0,0], 0, 0, explosion_image, explosion_info,explosion_sound)
            explosion_group.add(explosion)
            
            group.remove(g)
            return True
    return False

def group_group_collide(group1 , group2):
    
    global score;
    for g in list(group1):
        if group_collide(group2,g) :
            group1.discard(g)
            score +=1

def angle_to_vector(ang):
    return [math.cos(ang), math.sin(ang)]

def dist(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

###########################################
##
##	MAIN FUNCTION
##
###########################################	

if __name__ == '__main__':
    
	debris_info = ImageInfo([320, 240], [640, 480])
	debris_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/debris2_blue.png")


	nebula_info = ImageInfo([400, 300], [800, 600])
	nebula_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/nebula_blue.f2014.png")


	splash_info = ImageInfo([200, 150], [400, 300])
	splash_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/splash.png")


	ship_info = ImageInfo([45, 45], [90, 90], 35)
	ship_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/double_ship.png")


	missile_info = ImageInfo([5,5], [10, 10], 3, 50)
	missile_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/shot2.png")


	asteroid_info = ImageInfo([45, 45], [90, 90], 40)
	asteroid_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/asteroid_blue.png")


	explosion_info = ImageInfo([64, 64], [128, 128], 17, 24, True)
	explosion_image = simplegui.load_image("http://commondatastorage.googleapis.com/codeskulptor-assets/lathrop/explosion_alpha.png")


	soundtrack = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/soundtrack.mp3")
	missile_sound = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/missile.mp3")
	missile_sound.set_volume(.5)
	ship_thrust_sound = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/thrust.mp3")
	explosion_sound = simplegui.load_sound("http://commondatastorage.googleapis.com/codeskulptor-assets/sounddogs/explosion.mp3")

	frame = simplegui.create_frame("Asteroids", WIDTH, HEIGHT)

	my_ship = Ship([WIDTH / 2, HEIGHT / 2], [0, 0], 0, ship_image, ship_info)
	rock_group = set()
	missile_group = set()
	explosion_group = set()

	previous_closest_obstacle = None

	frame.set_keyup_handler(keyup)
	frame.set_keydown_handler(keydown)
	frame.set_mouseclick_handler(click)
	frame.set_draw_handler(draw)

	timer = simplegui.create_timer(1000.0, rock_spawner)


	timer.start()
	frame.start()

