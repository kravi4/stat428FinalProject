import random
import numpy as np 
import string


'''
first 8 - health
second 7 - poison radius
third 7 - food radius
fourth 6 - size
fifth 5 - food propensity
sixth 5 - poison propensity

'''

# compute the fitness score for each phrase
def compute_fitness(fish_list):
	fitness_list= []
	total_score = 0
	for fish in fish_list:
		# function we want to approximate
		score = fish.fitness*fish.fitness
		total_score+=score

		fitness_list.append((fish.dna, score))

	return fitness_list, total_score


# Create the selection process
def compute_selection(fitness_list, total_score):
	selection_pool = []
	
	# adds more phrases to the list based on the relative score
	for fit_fish in fitness_list:
		total_copy = int((float(fit_fish[1])/float(total_score))*len(fitness_list))
		for i in range(total_copy):
			selection_pool.append(fit_fish[0])

	return selection_pool


# recombine from selction process with mutation and return a dna bit string
def compute_recombination(selection_pool, mutation_factor = 0.01):

	top_2 = []
	
	for j in range(2):
		if j==0:
			top_2.append(random.choice(selection_pool))
			sub_pool = [ele for ele in selection_pool if ele != top_2[0]]
		else:
			if len(sub_pool) >0:
				top_2.append(random.choice(sub_pool))
			else:
				top_2.append(random.choice(selection_pool))

	# parent 1 attributes
	health_1 = top_2[0][0:8]
	poison_rad_1 = top_2[0][8:15]
	food_rad_1 = top_2[0][15:22]
	size_1 = top_2[0][22:28]
	food_prop_1 = top_2[0][28:33]
	poison_prop_1 = top_2[0][33:38]

	# parent 2 attributes
	health_2 = top_2[1][0:8]
	poison_rad_2 = top_2[1][8:15]
	food_rad_2 = top_2[1][15:22]
	size_2 = top_2[1][22:28]
	food_prop_2 = top_2[1][28:33]
	poison_prop_2 = top_2[1][33:38]

	parent_1 = [health_1, poison_rad_1, food_rad_1, size_1, food_prop_1, poison_prop_1]
	parent_2 = [health_2, poison_rad_2, food_rad_2, size_2, food_prop_2, poison_prop_2]

	new_dna_list = []

	for i in range(len(parent_1)):
		split_index = random.choice([i for i in range(len(parent_1[i]))])
		new_attribute = parent_1[i][0:split_index] + parent_2[i][split_index:]
		new_dna_list.append(new_attribute)


	# incorperates the mutation rate to alter the combined word
	for i in range(len(new_dna_list)):
		mutation_compound = int(mutation_factor*1000)
		mutation_list= [1]*mutation_compound + [0]*(1000-mutation_compound)
		determine_mutation = random.choice(mutation_list)
		if determine_mutation == 1:
			index_of_choice = random.choice([k for k in range(len(new_dna_list[i]))])
			if new_dna_list[i][index_of_choice] == "1":
				if index_of_choice != len(new_dna_list[i]) - 1:
					new_dna_list[i] = new_dna_list[i][0:index_of_choice] + "0" + new_dna_list[i][index_of_choice+1:]
				else:
					new_dna_list[i] = new_dna_list[i][0:index_of_choice] + "0"
			else:
				if index_of_choice != len(new_dna_list[i]) - 1:
					new_dna_list[i] = new_dna_list[i][0:index_of_choice] + "1" + new_dna_list[i][index_of_choice+1:]
				else:
					new_dna_list[i] = new_dna_list[i][0:index_of_choice] + "1"

	return new_dna_list
