from search import *
from random import randint
from assignment2aux import *
import copy

def read_tiles_from_file(filename):
    lines = [line.rstrip('\n') for line in open(filename, 'r').readlines()]
    character_to_tile = {' ': (), 'i': (0,), 'L': (0, 1), 'I': (0, 2), 'T': (0, 1, 2)}
    return tuple(tuple(character_to_tile[character] for character in line) for line in lines)
    

class KNetWalk(Problem):
    def __init__(self, tiles):
        if type(tiles) is str:
            self.tiles = read_tiles_from_file(tiles)
        else:
            self.tiles = tiles
        
        self.height = len(self.tiles)
        self.width = len(self.tiles[0])
        self.max_fitness = sum(sum(len(tile) for tile in row) for row in self.tiles)
        super().__init__(self.generate_random_state())

    def generate_random_state(self):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [randint(0, 3) for _ in range(height) for _ in range(width)]

    def actions(self, state):
        height = len(self.tiles)
        width = len(self.tiles[0])
        return [(i, j, k) for i in range(height) for j in range(width) for k in [0, 1, 2, 3] if state[i * width + j] != k]

    def result(self, state, action):
        pos = action[0] * len(self.tiles[0]) + action[1]
        return state[:pos] + [action[2]] + state[pos + 1:]

    def goal_test(self, state):
        return self.value(state) == self.max_fitness

    #self: initial state, state: a list of rotation instructions for each tile
    def value(self, state):
        # Task 1
        # Return an integer fitness value of a given state.
        # Replace the line below with your code.
        fitness = 0

        #create a copy of the state
        copied_self = copy.deepcopy(self)


        #create a list to store the row lists 
        map_list = []

        #loop thorough every tile on the board and rotate them
        for i in range(self.height):
            #list of rotated tile(tuple)
            row_list = []

            for j in range(self.width):
                #set current tile and grab rotation
                current_tile = self.tiles[i][j]
                rotation_needed = state[i * self.width + j]

                #rotate the tile and save it to a variable
                rotated_tile = tuple((con + rotation_needed) % 4 for con in current_tile)

                #set rotated tile back into the copied state
                row_list.append(rotated_tile)
            #adds row list into the map list
            map_list.append(row_list)
        
        for i in range(self.height):
            for j in range(self.width):
                #if tile to the right of current exists
                if j + 1 < self.width  :
                    #if the current tile has a right connection
                    if 0 in map_list[i][j] and 2 in map_list[i][j + 1]:
                        fitness += 1

                #if tile to the left of current exists
                if j - 1 >= 0:
                    #if the current tile has a left connection
                    if 2 in map_list[i][j] and 0 in map_list[i][j - 1]:
                        fitness += 1
                
                #if tile to the top of current exists
                if i - 1 >= 0:
                    #if the current tile has a top connection
                    if 1 in map_list[i][j] and 3 in map_list[i - 1][j]:
                        fitness += 1

                #if tile to the bottom of current exists
                if i + 1 < self.height:
                    #if the current tile has a bottom connection and bottom tile has a top connection
                    if 3 in map_list[i][j] and 1 in map_list[i + 1][j]:
                        fitness += 1
                
                
        return fitness
     


# Task 2
# Configure an exponential schedule for simulated annealing.

#k: the initial temprature (likelihood of choosing a bad state at the beginning)
#lam: the rate of decrease of temprature (how fast temp decreases)
#limit: max number of iterations
sa_schedule = exp_schedule(k=100, lam=1, limit=100)

# Task 3
# Configure parameters for the genetic algorithm.
pop_size = 13 #population size (bigger = better exploration but slower)
num_gen = 200 #no of generations (bigger = more runs)
mutation_prob = 0.15 #probablity of mutation per individual (randomness)

def local_beam_search(problem, population):
    # Task 4
    # Implement local beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the next population contains no fitter state.
    # Replace the line below with your code.

    #initialise current parent population
    parent_population = population  
    #beam width is size of initial population
    beam_width = len(population)

    while True:
        #initialise the next population
        children_population = []
        #for every tile in the board 
        for parent in parent_population:
            for action in problem.actions(parent):
                #get the child and add child into the children list
                child = problem.result(parent, action)
                children_population.append(child)
        
        #sort child from most fit to least fit
        children_population.sort(key=problem.value, reverse=True)
        #keep the b fittest children
        children_population = children_population[:beam_width]

        #return the state if a goal is found in the child list
        for state in children_population:
            if problem.goal_test(state):
                return state
        
        #gets the fittest of both parent and child generations            
        fittest_current = max(parent_population, key=problem.value)
        fittest_next = max(children_population, key=problem.value)

        #if next generation's fittest is worse or equal to current generation's fittest, return current generations fittest
        if problem.value(fittest_next) <= problem.value(fittest_current):
            return fittest_current
        
        #update the next generation to be the current child population
        parent_population = children_population
        



def stochastic_beam_search(problem, population, limit=1000):
    # Task 5
    # Implement stochastic beam search.
    # Return a goal state if found in the population.
    # Return the fittest state in the population if the generation limit is reached.
    # Replace the line below with your code.

    #initialise current parent population
    parent_population = population  
    #beam width is size of initial population
    beam_width = len(population)

    for i in range(limit):
        #initialise the next population
        children_population = []

        #for every tile in the board 
        for parent in parent_population:
            for action in problem.actions(parent):
                #get the child and add child into the children list
                child = problem.result(parent, action)
                children_population.append(child)

        #gets the fittest of both parent and child generations            
        fittest_current = max(parent_population, key=problem.value)
        fittest_next = max(children_population, key=problem.value)

        #return the state if a goal is found in the parent list
        for state in parent_population:
            if problem.goal_test(state):
                return state

        #return the state if a goal is found in the child list
        for state in children_population:
            if problem.goal_test(state):
                return state
        
        #if next generation's fittest is worse or equal to current generation's fittest, return current generations fittest
        if problem.value(fittest_next) <= problem.value(fittest_current):
            return fittest_current
        else:
            #sum of all fitness in the current generation
            fitness_sum = 0
            #list of all probability for each child
            prob_list = []

            index_list = []
            iterationCounter = 0
            #gets the sum of all fitness
            for state in children_population:
                fitness_sum = problem.value(state) + fitness_sum
                index_list.append(iterationCounter)
                iterationCounter = iterationCounter + 1
        
            #gathers the probability for each children and put it into the list
            for state in children_population:
                prob_list.append(problem.value(state)/fitness_sum)

            #keep the b fittest via their probability
            selected_children_indexes = np.random.choice(index_list, beam_width, False, prob_list)

            #create the list and stores the selected children to expand
            selected_children = []
            for index in selected_children_indexes:
                selected_children.append(children_population[index])
        
            #update the next generation to be the current child population
            parent_population = selected_children



if __name__ == '__main__':

    network = KNetWalk('assignment2config.txt')
    visualise(network.tiles, network.initial)

    # Task 1 test code

    run = 0
    method = 'hill climbing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = hill_climbing(network)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 2 test code
    
    run = 0
    method = 'simulated annealing'
    while True:
        network = KNetWalk('assignment2config.txt')
        state = simulated_annealing(network, schedule=sa_schedule)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 3 test code
    
    run = 0
    method = 'genetic algorithm'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = genetic_algorithm([network.generate_random_state() for _ in range(pop_size)], network.value, [0, 1, 2, 3], network.max_fitness, num_gen, mutation_prob)
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 4 test code
    
    run = 0
    method = 'local beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = local_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    

    # Task 5 test code
    
    run = 0
    method = 'stochastic beam search'
    while True:
        network = KNetWalk('assignment2config.txt')
        height = len(network.tiles)
        width = len(network.tiles[0])
        state = stochastic_beam_search(network, [network.generate_random_state() for _ in range(100)])
        if network.goal_test(state):
            break
        else:
            print(f'{method} run {run}: no solution found')
            print(f'best state fitness {network.value(state)} out of {network.max_fitness}')
            visualise(network.tiles, state)
        run += 1
    print(f'{method} run {run}: solution found')
    visualise(network.tiles, state)
    
 