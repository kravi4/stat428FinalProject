# Library imports

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ListProperty, StringProperty, ObjectProperty, NumericProperty, ReferenceListProperty, DictProperty, BooleanProperty
from kivy.lang import Builder
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.graphics import *
from genetic_algorithm import *
from random import randint
from random import randrange
from copy import deepcopy
import numpy as np
import math
from math import ceil
from math import acos
from math import sqrt
from math import pi
from scipy.stats import norm
from scipy import stats
import random


# loads in the Kivy file
Builder.load_file('fish.kv')


# Global Constants for the Radii of the food and the poison
POISON_RADIUS = 10
FOOD_RADIUS = 10

# Global Constant for the number of fishes
NUM_FISHES = 20

# Function to compute Distance between two points (coordinates)
def compute_distance(center_point, dest_point):
    distance = np.sqrt(np.square(
        dest_point[0] - center_point[0]) + np.square(dest_point[1] - center_point[1]))
    return distance


def normalize_magnitude(vector):
    magnitude = vector[0]**2 + vector[1]**2
    magnitude = magnitude**0.5
    if magnitude != 0:
        vector = Vector(vector) * (1 / magnitude)
    return vector


# a = 2 b = 5
# Target Function
def kamaruswamy(x,a,b):
    if x < 1 and x > 0:
        return a*b*(x**(a-1))*(1-(x**a))**(b-1)

# Helper Factorial Function
def factorial(n):
    return (n/math.e) * math.sqrt(math.pi *(2*n + (1/3)))

# a= 2.71 b = 5.39
# Proposal Function
def beta(x,a,b):
    gamma = (factorial(a-1) * factorial(b-1))/factorial(a + b -1)
    pdf   = ((x**(a-1))*((1-x)**(b-1)))/gamma
    return pdf

# accept reject algorithm
def accept_reject():
    m = .646
    k = 0 # number accepted
    i = 1000 # number iterations
    n = 6
    saved = []

    while k < n:
        u = random.random()
        y = random.random()

        if u*m < kamaruswamy(y,2.5,5.17)/beta(y,1.97,6.6):
            k += 1
            saved.append(y)
    return saved

# Class for the Fish
class Fish(Widget):

    # Color values designated for health
    colors = ListProperty([[255 / 255, 0, 0, 1], [255 / 255, 43 / 255, 0, 1], [255 / 255, 85 / 255, 0, 1], [255 / 255, 128 / 255, 0, 1], [255 / 255, 170 / 255, 0, 1], [255 / 255, 213 / 255, 0, 1], [
                          255 / 255, 255 / 255, 0, 1], [213 / 255, 255 / 255, 0, 1], [170 / 255, 255 / 255, 0, 1], [128 / 255, 255 / 255, 0, 1], [85 / 255, 255 / 255, 0, 1], [43 / 255, 255 / 255, 0, 1], [0, 255 / 255, 0, 1]])
    color_val = ListProperty([0, 255 / 255, 0, 1])
    color_index = NumericProperty(12)

    # You'll have to change these later to be dynamic
    health = NumericProperty(0)
    colorLowBound = NumericProperty(0)

    # Variables designated for movement (not to be determined from the DNA)
    angle_val = NumericProperty(1)
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    max_velocity = NumericProperty(2)
    max_force = NumericProperty(0.5)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    acceleration_x = NumericProperty(0)
    acceleration_y = NumericProperty(0)
    acceleration = ReferenceListProperty(acceleration_x, acceleration_y)

    # Variables to be determined by the DNA when implemented
    poison_rad = NumericProperty(0)
    food_rad = NumericProperty(0)
    size_val = NumericProperty(0)
    food_propensity = NumericProperty(0)
    poison_propensity = NumericProperty(0)

    # objective fitness
    fitness = NumericProperty(0)

    # DNA once the variables above have been initialized
    dna = StringProperty('')

    # Function to initialize the poison perception radius, food perception radius and the size of the fish (used for the first randomization)
    def initialize_beginning(self):

        # initializing random values for the health, poison_rad, food_rad, size, and the food and poison propensity
        samples = accept_reject()
        health_val = samples[0] * 201
        health_val = ceil(health_val)
        if health_val > 200:
            health_val = 200
        elif health_val < 20:
            health_val = 20

        poison_rad = samples[1] * 128
        poison_rad = ceil(poison_rad)
        if poison_rad > 127:
            poison_rad = 127
        elif poison_rad < 1:
            poison_rad = 1

        food_rad = samples[2] * 128
        food_rad = ceil(food_rad)
        if food_rad > 127:
            food_rad = 127
        elif food_rad < 1:
            food_rad = 1

        size_val = samples[3] * 64
        size_val = ceil(size_val)
        if size_val > 63:
            size_val = 63
        elif size_val < 1:
            size_val = 1

        food_prop = samples[4] * 16
        food_prop = ceil(food_prop)
        if food_prop > 15:
            food_prop = 15
        elif food_prop < 1:
            food_prop = 1

        poison_prop = samples[4] * 16
        poison_prop = ceil(poison_prop)
        if poison_prop > 15:
            poison_prop = 15
        elif poison_prop < 1:
            poison_prop = 1

        self.health = health_val
        self.colorLowBound = self.health - self.health / 13
        self.poison_rad = poison_rad
        self.food_rad = food_rad
        self.size_val = size_val
        self.food_propensity = 1 / food_prop
        self.poison_propensity = -1 / poison_prop

        attributes = [self.health, self.poison_rad,
                      self.food_rad, self.size_val, food_prop, poison_prop]

        # Generates the DNA string from the initialized values
        final_dna_string = ""
        for i in range(len(attributes)):
            dna_bitstring_val = ""
            if i == 0:
                dna_bitstring_val = '{0:08b}'.format(attributes[i])
            elif i == 1 or i == 2:
                dna_bitstring_val = '{0:07b}'.format(attributes[i])
            elif i == 3:
                dna_bitstring_val = '{0:06b}'.format(attributes[i])
            else:
                if attributes[i] < 0:
                    attributes[i] *= -1
                    dna_bitstring_val = "1" + '{0:04b}'.format(attributes[i])
                else:
                    dna_bitstring_val = "0" + '{0:04b}'.format(attributes[i])

            final_dna_string += dna_bitstring_val

        self.dna = final_dna_string

    # TODO

    def initialize_from_DNA(self, dna_list):

        # initializing the health
        if dna_list[0] == "00000000":
            self.health = 1
            dna_list[0] = "00000001"
        else:
            self.health = int(dna_list[0], 2)

        # Initializing the lower color bound
        self.colorLowBound = self.health - self.health / 13

        # Initializing the poison radius
        if dna_list[1] == "0000000":
            self.poison_rad = 1
            dna_list[1] = "0000001"
        else:
            self.poison_rad = int(dna_list[1], 2)

        # Initializing the food radius
        if dna_list[2] == "0000000":
            self.food_rad = 1
            dna_list[2] = "0000001"
        else:
            self.food_rad = int(dna_list[2], 2)

        # Initializing the size
        if dna_list[3] == "000000":
            self.size_val = 1
            dna_list[3] = "000001"
        else:
            self.size_val = int(dna_list[3], 2)

        # Initializing the food perception
        if dna_list[4][0] == 1:
            if dna_list[4][1:] == "0000":
                self.food_propensity = -1
                dna_list[4] = "10001"
            else:
                self.food_propensity = -1 / int(dna_list[4][1:], 2)
        else:
            if dna_list[4][1:] == "0000":
                self.food_propensity = 1
                dna_list[4] = "00001"
            else:
                self.food_propensity = 1 / int(dna_list[4][1:], 2)

        # Initializing the poison perception
        if dna_list[5][0] == 1:
            if dna_list[5][1:] == "0000":
                self.poison_propensity = -1
                dna_list[5] = "10001"
            else:
                self.poison_propensity = -1 / int(dna_list[5][1:], 2)
        else:
            if dna_list[5][1:] == "0000":
                self.poison_propensity = 1
                dna_list[5] = "00001"
            else:
                self.poison_propensity = 1 / int(dna_list[5][1:], 2)

        for attr in dna_list:
            self.dna += attr

    # adjusts the steering force based on the target position of the food or the poison
    def seek(self, current_pos, target_pos):
        desired_velocity = Vector(target_pos) - Vector(current_pos)
        desired_velocity = normalize_magnitude(
            desired_velocity) * self.max_velocity
        steering_force = desired_velocity - Vector(self.velocity)
        steering_force = normalize_magnitude(steering_force) * self.max_force
        return(steering_force)

    # Applies the force to the acceleration
    def applyForce(self, force):
        # print(force)
        self.acceleration = Vector(self.acceleration) + force

    # Function to move the car
    def move(self):
        # print(self.velocity)
        self.velocity = Vector(self.velocity) + self.acceleration
        self.velocity = Vector(normalize_magnitude(
            self.velocity)) * self.max_velocity
        self.pos = Vector(*self.velocity) + self.pos
        self.acceleration = Vector(self.acceleration) * 0

        # Adjusts the color as the health of the car decreases
        if self.health < self.colorLowBound and self.color_index >= 0:
            self.color_val = self.colors[self.color_index]
            self.color_index -= 1
            self.colorLowBound -= self.health / 13

# Class designed for the car's poison radius, this is needed for displying the poison radius of the fish on the gui


class PoisonRadius(Widget):

    # the Center x and y position as well as the radius value
    cx = NumericProperty(0)
    cy = NumericProperty(0)
    radius = NumericProperty(0)

    # Function to initialize the the properties
    def initialize(self, x, y, rad):
        self.cx = x
        self.cy = y
        self.radius = rad

    # Function to move the Poison Radius
    def move(self, x, y):
        self.cx = x
        self.cy = y


# Class designed for the car's food radius, this is needed for displying the poison radius of the fish on the gui
class FoodRadius(Widget):

    # the Center x and y position as well as the radius value
    cx = NumericProperty(0)
    cy = NumericProperty(0)
    radius = NumericProperty(0)

    # Function to initialize the the properties
    def initialize(self, x, y, rad):
        self.cx = x
        self.cy = y
        self.radius = rad

    # Function to move the Food Radius
    def move(self, x, y):
        self.cx = x
        self.cy = y


''' The two classes below are placeholders that will be inherently invoked through the kivy file '''
class Food(Widget):
    pass


class Poison(Widget):
    pass


# This class will be responsible for actually running the simulation itself
class FishSimulation(Widget):

    fish_fitness_gen = []
    fish_fitness_over_gens = []
    fish_list = ListProperty([])
    fish_list_genetic_pool = ListProperty([])
    food_list = ListProperty([])
    poison_list = ListProperty([])
    food_dict = DictProperty({})
    poison_dict = DictProperty({})
    gen_val = StringProperty("Generation 1")
    gen_count = NumericProperty(1)
    count = NumericProperty(0)
    start_new_gen = BooleanProperty(False)
    avg_fitness = NumericProperty(0)

    # initial population of fishes and food and poison in the first generation
    def populate_fishes_and_food(self):

        for i in range(NUM_FISHES):
            fish = Fish()
            fish.initialize_beginning()
            poison_radius = PoisonRadius()
            food_radius = FoodRadius()
            poison_radius.initialize(
                fish.center_x, fish.center_y, fish.poison_rad)
            food_radius.initialize(fish.center_x, fish.center_y, fish.food_rad)
            fish.add_widget(poison_radius)
            fish.add_widget(food_radius)
            self.add_widget(fish)
            self.fish_list.append(fish)
            self.fish_list_genetic_pool.append(fish)

        for i in range(int(NUM_FISHES * 5)):
            food = Food()
            poison = Poison()

            food_x, food_y = randint(0, 800), randint(0, 600)
            poison_x, poison_y = randint(0, 800), randint(0, 600)

            while poison_x == food_x and poison_y == food_y:
                poison_x, poison_y = randint(0, 800), randint(0, 600)

            food_coord = food_x, food_y
            poison_coord = poison_x, poison_y

            food.pos = food_coord
            poison.pos = poison_coord

            self.add_widget(food)
            self.add_widget(poison)

            self.food_list.append(food)
            self.poison_list.append(poison)

            self.food_dict[food_coord] = food
            self.poison_dict[poison_coord] = poison

    # populating the fishes from the genetic algorithm
    def populate_fishes_from_genetic_algorithm(self):

        fitness_list, total_score = compute_fitness(
            self.fish_list_genetic_pool)
        selection_pool = compute_selection(fitness_list, total_score)
        self.fish_list_genetic_pool = []
        for i in range(NUM_FISHES):
            new_dna_list = compute_recombination(selection_pool)
            fish = Fish()
            fish.initialize_from_DNA(new_dna_list)
            poison_radius = PoisonRadius()
            food_radius = FoodRadius()
            poison_radius.initialize(
                fish.center_x, fish.center_y, fish.poison_rad)
            food_radius.initialize(fish.center_x, fish.center_y, fish.food_rad)
            fish.add_widget(poison_radius)
            fish.add_widget(food_radius)
            self.add_widget(fish)
            self.fish_list.append(fish)
            self.fish_list_genetic_pool.append(fish)

    # Populating the food and the poison on the sceen
    def populate_food_and_poison(self):

        for child in self.children:
            self.remove_widget(child)

        self.food_list = []
        self.poison_list = []
        self.food_dict = {}
        self.poison_dict = {}

        for i in range(int(NUM_FISHES * 5)):
            food = Food()
            poison = Poison()

            food_x, food_y = randint(0, 800), randint(0, 600)
            poison_x, poison_y = randint(0, 800), randint(0, 600)

            while poison_x == food_x and poison_y == food_y:
                poison_x, poison_y = randint(0, 800), randint(0, 600)

            food_coord = food_x, food_y
            poison_coord = poison_x, poison_y

            food.pos = food_coord
            poison.pos = poison_coord

            self.add_widget(food)
            self.add_widget(poison)

            self.food_list.append(food)
            self.poison_list.append(poison)

            self.food_dict[food_coord] = food
            self.poison_dict[poison_coord] = poison

    # function that initializes the fishes' initial velocity and direction
    def serve_fishes(self):

        for fish in self.fish_list:
            fish.center = self.center_x + \
                randint(-200, 200), self.center_y + randint(-200, 200)
            fish.velocity = Vector(randrange(-fish.max_velocity, fish.max_velocity),
                                   randrange(-fish.max_velocity, fish.max_velocity))

    # Function that allows fish to eat poison
    def eat_food(self, fish, food_count, food_in_radius, current_pos):

        for coord in self.food_dict:
            if fish.size_val < FOOD_RADIUS:
                if (current_pos[0] <= coord[0] + 10 and current_pos[0] >= coord[0] - 10) and (current_pos[1] <= coord[1] + 10 and current_pos[1] >= coord[1] - 10):
                    food_coord = randint(0, 800), randint(0, 600)

                    while food_coord in self.food_dict or food_coord in self.poison_dict:
                        food_coord = randint(0, 800), randint(0, 600)

                    new_food = Food()
                    new_food.pos = food_coord
                    self.remove_widget(self.food_dict[coord])
                    self.food_dict.pop(coord, None)
                    self.add_widget(new_food)
                    self.food_dict[food_coord] = new_food

                    fish.health = fish.health + 1
                    fish.fitness += 1
            else:
                if ((coord[0] - current_pos[0])**2 + (coord[1] - current_pos[1])**2) <= (fish.size_val / 2)**2:
                    food_coord = randint(0, 800), randint(0, 600)

                    while food_coord in self.food_dict or food_coord in self.poison_dict:
                        food_coord = randint(0, 800), randint(0, 600)

                    new_food = Food()
                    new_food.pos = food_coord
                    self.remove_widget(self.food_dict[coord])
                    self.food_dict.pop(coord, None)
                    self.add_widget(new_food)
                    self.food_dict[food_coord] = new_food

                    fish.health = fish.health + 1
                    fish.fitness += 1

            if coord in self.food_dict:
                distance = compute_distance(current_pos, coord)
                if distance <= fish.food_rad:
                    food_in_radius[distance] = coord
                    food_count += 1

            if len(list(food_in_radius.keys())) > 0:
                min_food_dist = min(list(food_in_radius.keys()))
            else:
                min_food_dist = 0

            if min_food_dist != 0:
                min_food_coord = food_in_radius[min_food_dist]

            if food_count > 0:
                seek = fish.seek(current_pos, min_food_coord)
                seek *= fish.food_propensity
                seek = normalize_magnitude(seek) * fish.max_force
                fish.applyForce(seek)

    # Function that allows fish to eat poison
    def eat_poison(self, fish, poison_count, poison_in_radius, current_pos):

        for coord in self.poison_dict:
            if fish.size_val < POISON_RADIUS:
                if (current_pos[0] <= coord[0] + 5 and current_pos[0] >= coord[0] - 5) and (current_pos[1] <= coord[1] + 5 and current_pos[1] >= coord[1] - 5):
                    poison_coord = randint(0, 800), randint(0, 600)

                    while poison_coord in self.food_dict or poison_coord in self.poison_dict:
                        poison_coord = randint(0, 800), randint(0, 600)

                    new_poison = Poison()
                    new_poison.pos = poison_coord
                    self.remove_widget(self.poison_dict[coord])
                    self.poison_dict.pop(coord, None)
                    self.add_widget(new_poison)
                    self.poison_dict[poison_coord] = new_poison

                    fish.health = fish.health * 0.5

            else:
                if ((coord[0] - current_pos[0])**2 + (coord[1] - current_pos[1])**2) <= (fish.size_val / 2)**2:
                    poison_coord = randint(0, 800), randint(0, 600)

                    while poison_coord in self.food_dict or poison_coord in self.poison_dict:
                        poison_coord = randint(0, 800), randint(0, 600)

                    new_poison = Poison()
                    new_poison.pos = poison_coord
                    self.remove_widget(self.poison_dict[coord])
                    self.poison_dict.pop(coord, None)
                    self.add_widget(new_poison)
                    self.poison_dict[poison_coord] = new_poison

                    fish.health = fish.health * 0.5

            if coord in self.poison_dict:
                distance = compute_distance(current_pos, coord)
                if distance < fish.poison_rad:
                    poison_in_radius[distance] = coord
                    poison_count += 1

            if len(list(poison_in_radius.keys())) > 0:
                min_poison_dist = min(list(poison_in_radius.keys()))
            else:
                min_poison_dist = 0

            if min_poison_dist != 0:
                min_poison_coord = poison_in_radius[min_poison_dist]

            if poison_count > 0:
                seek = fish.seek(current_pos, min_poison_coord)
                seek *= fish.poison_propensity
                seek = normalize_magnitude(seek) * fish.max_force
                fish.applyForce(seek)

    # Function for bootstrap
    def bootstrap(self):

        n = len(self.fish_fitness_gen)

        B = 10000
        theta_hat = np.mean(self.fish_fitness_gen)
        Tboot = [0 for b in range(B)]
        for b in range(B):
            xb = np.random.choice(
                self.fish_fitness_gen, n, replace=True)
            Tboot[b] = np.mean(xb)
        se_theta_hat = np.std(Tboot)
        ci_upper = theta_hat + norm.ppf(.975) * se_theta_hat
        ci_lower = theta_hat - norm.ppf(.975) * se_theta_hat
        print('mean estimate: ' + str(theta_hat))
        print('bootstrap se of estimate: ' + str(se_theta_hat))
        print('bootstrap 95% ci: ' +
              '(' + str(ci_lower) + ', ' + str(ci_upper) + ')')

    # Function for jackknife
    def jackknife(self):
        n = len(self.fish_fitness_gen)

        theta_hat = np.mean(self.fish_fitness_gen)
        thetahatjack = [0 for b in range(n)]
        for i in range(n):
            temp = self.fish_fitness_gen[i]
            del self.fish_fitness_gen[i]
            thetahatjack[i] = np.mean(self.fish_fitness_gen)
            self.fish_fitness_gen.insert(i, temp)

        sumsq = sum(np.square(np.subtract(
            thetahatjack, np.mean(thetahatjack))))
        se_jackknife = sqrt((n - 1) / n) * sqrt(sumsq)
        bias_jackknife = (n - 1) * (np.mean(thetahatjack) - theta_hat)
        print('jackknife se of estimate: ' + str(se_jackknife))
        print('jackknife bias of estimate: ' + str(bias_jackknife))
        print()

    # function for the permutation test
    def permutationTest(self):
        B = 10000

        x = self.fish_fitness_over_gens[-1]
        y = self.fish_fitness_over_gens[-2]
        z = x + y

        nu = np.arange(len(z))
        reps = [0 for b in range(B)]
        t0 = abs(stats.ttest_ind(x, y, equal_var=False)[0])
        print('initial t: ' + str(t0))

        for i in range(B):
            perm = np.random.choice(nu, len(x), replace=True)
            x1 = []
            for j in perm:
                x1.append(z[j])

            y1 = []
            for j in [item for item in nu if item not in set(perm)]:
                y1.append(z[j])

            reps[i] = abs(stats.ttest_ind(x1, y1, equal_var=False)[0])

        reps.append(t0)
        greater_than_t0 = 0
        for i in reps:
            if i >= t0:
                greater_than_t0 += 1
        p_value = greater_than_t0 / len(reps)
        print('computed p_value: ' + str(p_value))
        print()

    # update function to update on each time step
    def update(self, dt):
        count = 1

        if self.start_new_gen == True:
            self.gen_count += 1
            self.gen_val = "Generation " + str(self.gen_count)
            self.populate_food_and_poison()
            self.populate_fishes_from_genetic_algorithm()
            self.serve_fishes()
            self.start_new_gen = False

        else:
            for fish in self.fish_list:
                food_in_radius = {}
                poison_in_radius = {}

                angle_movement = 0

                current_pos = fish.center_x, fish.center_y
                food_count = 0
                poison_count = 0
                # print(current_pos)
                self.eat_food(fish, food_count, food_in_radius, current_pos)
                self.eat_poison(fish, poison_count,
                                poison_in_radius, current_pos)

                # bounce off top and bottom
                if fish.y < 0:
                    fish.y = 0
                    fish.velocity_y *= -1

                elif fish.top > self.height:
                    fish.top = self.height
                    fish.velocity_y *= -1

                if (fish.x < 0):
                    fish.x = 0
                    fish.velocity_x *= -1

                elif (fish.right > self.width):
                    fish.right = self.width
                    fish.velocity_x *= -1

                fish.move()
                for rad in fish.children:
                    rad.move(fish.center_x, fish.center_y)

                if fish.health > 200:
                    fish.health = 200

                fish.health -= 0.2

                if fish.health <= 0:
                    fish.health = 0
                    self.avg_fitness += fish.fitness
                    self.fish_fitness_gen.append(fish.fitness)
                    for rad in fish.children:
                        fish.remove_widget(rad)

                    self.remove_widget(fish)
                    self.fish_list.remove(fish)

            if len(self.fish_list) == 0:
                self.fish_fitness_over_gens.append(self.fish_fitness_gen)
                self.bootstrap()
                self.jackknife()
                if self.gen_count > 1:
                    self.permutationTest()

                self.avg_fitness = self.avg_fitness / \
                    len(self.fish_list_genetic_pool)
                self.avg_fitness = 0
                self.start_new_gen = True
                self.fish_fitness_gen = []


# class that runs the simulation
class FishApp(App):
    def build(self):
        game = FishSimulation()
        game.populate_fishes_and_food()
        game.serve_fishes()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game


if __name__ == '__main__':
    FishApp().run()
