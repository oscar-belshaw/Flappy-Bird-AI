import pygame
import neat
import time
import os
import random
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 700
FLLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

#defines the window
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

#Imports the images to be used
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

gen = 0

STAT_FONT = pygame.font.SysFont("comicsans", 50, bold=True)

#defines the bird and how it moves, along with the jump function
class Bird:
	IMGS = BIRD_IMGS
	MAX_ROTATION = 25
	ROTATION_VEL = 20
	ANIMATION_TIME = 3

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.tilt = 0
		self.tick_count = 0
		self.velocity = 0
		self.height = self.y
		self.img_count = 0
		self.img =  self.IMGS[0]

	def jump(self):
		 self.velocity = -10.5
		 self.tick_count = 0
		 self.height = self.y

	def move(self):
		self.tick_count += 1

		d = self.velocity*self.tick_count + 1.5*self.tick_count**2

		if d >= 16:
			d = 16
		elif d <0:
			d -= 2
		self.y = self.y + d

		if d < 0 or self.y < self.height + 50:
			if self.tilt < self.MAX_ROTATION:
				self.tilt = self.MAX_ROTATION

		else:
			if self.tilt > -90:
				self.tilt -= self.ROTATION_VEL

	def draw(self, win):
		self.img_count += 1

		if self.img_count < self.ANIMATION_TIME:
			self.img = self.IMGS[0]
		elif self.img_count < self.ANIMATION_TIME*2:
			self.img = self.IMGS[1]
		elif self.img_count < self.ANIMATION_TIME*3:
			self.img = self.IMGS[2]
		elif self.img_count < self.ANIMATION_TIME*4:
			self.img = self.IMGS[1]
		elif self.img_count == self.ANIMATION_TIME*4 + 1:
			self.img = self.IMGS[0]
			self.img_count = 0

		if self.tilt <= -80:
			self.img = self.IMGS[1]
			self.img_count = self.ANIMATION_TIME*2

		rotated_image = pygame.transform.rotate(self.img, self.tilt)
		new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
		win.blit(rotated_image, new_rect.topleft)
	
	def get_mask(self):
		return pygame.mask.from_surface(self.img)


#Defines the pipe class along with the collisions between a pipe and a bird
#which causes a game over
class Pipe():
	GAP = 200
	VELOCITY = 5

	def __init__(self, x):
		self.x = x
		self.height = 0 

		self.top = 0
		self.bottom = 0
		self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
		self.PIPE_BTM = PIPE_IMG

		self.passed = False

		self.set_height()

	def set_height(self):
		self.height = random.randrange(50, 450)
		self.top = self.height - self.PIPE_TOP.get_height()
		self.bottom = self.height + self.GAP

	def move(self):
		self.x -= self.VELOCITY

	def draw(self, win):
		win.blit(self.PIPE_TOP, (self.x, self.top))
		win.blit(self.PIPE_BTM, (self.x, self.bottom))

	#The collision finction handles when a bird hits a pipe ie they are at the
	#same coordinates
	def collide(self, bird):
		bird_mask = bird.get_mask()
		top_mask = pygame.mask.from_surface(self.PIPE_TOP)
		bottom_mask = pygame.mask.from_surface(self.PIPE_BTM)

		top_offset = (self.x - bird.x, self.top - round(bird.y))
		bottom_offset = (self.x -bird.x, self.bottom - round(bird.y))

		b_point = bird_mask.overlap(bottom_mask, bottom_offset)
		t_point = bird_mask.overlap(top_mask, top_offset)

		if t_point or b_point:
			return True

		return False

#This class defines the base, which is also a collision surface
class Base:
	VELOCITY = 5
	WIDTH = BASE_IMG.get_width()
	IMG = BASE_IMG

	def __init__(self, y):
		self.y = y
		self.x1 = 0
		self.x2 = self.WIDTH

	#The base is constantly moving horizontally to give the illusion of movement
	def move(self):
		self.x1 -= self.VELOCITY
		self.x2 -= self.VELOCITY

		if self.x1 + self.WIDTH < 0:
			self.x1 = self.x2 + self.WIDTH
		if self.x2 + self.WIDTH < 0:
			self.x2 = self.x1 + self.WIDTH
	def draw(self, win):
		 win.blit(self.IMG, (self.x1, self.y))
		 win.blit(self.IMG, (self.x2, self.y))

#Draws everything on screen
def draw_window(win, birds, pipes, base, score):
	if gen == 0:
		gen = 1
	win.blit(BG_IMG, (0,0))
	for pipe in pipes:
		pipe.draw(win)

	text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
	win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

	base.draw(win)
	for bird in birds: 
		bird.draw(win)
	pygame.display.update()

#The main function brings together all game functionality and
#runs the machine learning algorithm
def main(genomes, config):
	nets = []
	ge = []
	birds = []

	#Initialising the ML
	for _, g in genomes:
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)
		birds.append(230, 350)
		g.fitness = 0
		ge.append(g)

	#Initialising the game
	base = Base(630)
	pipes = [Pipe(600)]
	win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT)) 
	clock = pygame.time.Clock()

	score = 0

	#This is where the game loop starts
	run = True
	while run:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()

			pipe_ind = 0

			if len(birds) > 0:
				if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
					pipe_ind = 1

			else:
				run = False
				break 

			for x, bird in enumerate(birds):
				bird.move()
				ge[x].fitness += 0.1

				output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

				if output[0] > 0.5:
					bird.jump()

			bird.move()
		add_pipe = False 
		remove = []
		for pipe in pipes:
			for x, bird in enumerate(birds):
				#This tells the algorithm that a bird has collided with a pipe
				if pipe.collide(bird):
					ge[x].fitness -= 1
					birds.remove(bird)
					birds.pop(x)
					nets.pop(x)
					ge.pop(x)
					

				if not pipe.passed and pipe.x < bird.x:
					pipe.passed = True
					add_pipe = True

			if pipe.x + pipe.PIPE_TOP.get_width() < 0:
				remove.append(pipe)


			pipe.move()

		#When a pipe is passed a new pipe is added and score is increased by 1
		if add_pipe:
			score += 1 
			for g in ge:
				g.fitness += 5
			pipes.append(Pipe(600))

		#The pipe is then removed
		for r in remove: 
			pipes.remove(r)
			pipe.move()

		#If the bird goes off screen it is removed from the algorithm
		for x, bird in enumerat(birds):
			if bird.y + bird.img.get_height() >=  630 or bird.y < 0:
				 birds.pop(x)
				 nets.pop(x)
				 ge.pop(x)
		
		base.move()
		draw_window(win, birds, pipes, base, score, gen, pipe_ind) 

#This runs the ML algorithm from the config file
def run(config_file):
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
							neat.DefaultSpeciesSet, neat.DefaultStagnation,
							config_file)

	population = neat.Population(config)

	population.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	population.add_reporter(stats)


if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-feedforward.txt")
	run(config_path)


