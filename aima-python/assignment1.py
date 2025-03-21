from time import time
from search import *
from assignment1aux import *

def read_initial_state_from_file(filename):
    # Task 1
    # Return an initial state constructed using a configuration in a file.
    # Replace the line below with your code.
    # raise NotImplementedError

    #open the file to read
    config_file = open(filename)

    # gets height of the map
    height = int(config_file.readline())
    

    #gets width of the map
    width = int(config_file.readline())
 

    #stores rock coords
    rock_coordinates_list = []

    #loops thorugh the file
    for line in config_file:
        #casts the numbers from string to int as it gets read and puts it into a list
        current_rock_coords = [int (value) for value in line.strip().split(',')]

    
        #add to list of rocks
        rock_coordinates_list.append(current_rock_coords)

    #prints out the rock coordinates
    print(rock_coordinates_list)

    #creates map as a list which will later be converted into a tuple
    map = []

    #creates the 2d map
    for row in range(height):

        #creates a list that represent the current row
        row_list = []
        
        
        for col in range(width):

            #set bool to see if current coordinate has a rock
            rock_found = False


            #loop through the list of rocks to compare with current coordinate to see if it matches
            for rock_coordinates in rock_coordinates_list:


                
                if rock_coordinates[0] == row and rock_coordinates[1] == col:
                    
                    #if coordinate matches add rock into current coordinate
                    row_list.append("rock")
                    rock_found = True
                    break   

            #if rock is not found, add a space into the current coordinate
            if not rock_found:
                    row_list.append('')


        
        #add the current list of coordinate into the map and convert into tuple
        map.append(tuple(row_list))


    #convert map into tuple then print out for checking
    map = tuple(map)
    for row in map:
        print(row)
    
    
    
    
     #close the file
    config_file.close()

    #returns the state with the map and two other conditions thats currently null
    state = (map, None, None)
    return state
                    

   

class ZenPuzzleGarden(Problem):
    def __init__(self, initial):
        if type(initial) is str:
            super().__init__(read_initial_state_from_file(initial))
        else:
            super().__init__(initial)

    def actions(self, state):
        map = state[0]
        position = state[1]
        direction = state[2]
        height = len(map)
        width = len(map[0])
        action_list = []
        if position:
            if direction in ['up', 'down']:
                if position[1] == 0 or not map[position[0]][position[1] - 1]:
                    action_list.append((position, 'left'))
                if position[1] == width - 1 or not map[position[0]][position[1] + 1]:
                    action_list.append((position, 'right'))
            if direction in ['left', 'right']:
                if position[0] == 0 or not map[position[0] - 1][position[1]]:
                    action_list.append((position, 'up'))
                if position[0] == height - 1 or not map[position[0] + 1][position[1]]:
                    action_list.append((position, 'down'))
        else:
            for i in range(height):
                if not map[i][0]:
                    action_list.append(((i, 0), 'right'))
                if not map[i][width - 1]:
                    action_list.append(((i, width - 1), 'left'))
            for i in range(width):
                if not map[0][i]:
                    action_list.append(((0, i), 'down'))
                if not map[height - 1][i]:
                    action_list.append(((height - 1, i), 'up'))
        return action_list

    def result(self, state, action):
        map = [list(row) for row in state[0]]
        position = action[0]
        direction = action[1]
        height = len(map)
        width = len(map[0])
        while True:
            row_i = position[0]
            column_i = position[1]
            if direction == 'left':
                new_position = (row_i, column_i - 1)
            if direction == 'up':
                new_position = (row_i - 1, column_i)
            if direction == 'right':
                new_position = (row_i, column_i + 1)
            if direction == 'down':
                new_position = (row_i + 1, column_i)
            if new_position[0] < 0 or new_position[0] >= height or new_position[1] < 0 or new_position[1] >= width:
                map[row_i][column_i] = direction
                return tuple(tuple(row) for row in map), None, None
            if map[new_position[0]][new_position[1]]:
                return tuple(tuple(row) for row in map), position, direction
            map[row_i][column_i] = direction
            position = new_position

    def goal_test(self, state):
        # Task 2
        # Return a boolean value indicating if a given state is solved.
        # Replace the line below with your code.
        
        #stores the boolean values of the loop that checks for empty spaces in the state
        list_for_checking = []

        #loops through the state index 0 (the map) 
        for row in state[0]:
           #using the all() to see if the row contains any blank spaces then adds to the list for checking
           list_for_checking.append(all(row)) 

        #runs all() to check if any value is false return false if so
        return all(list_for_checking)

       


# Task 3
# Implement an A* heuristic cost function and assign it to the variable below.
def astar_heuristic_cost(node):
    
    #gets the map from the state
    map = node.state[0]

    #gets the height and width of the map
    height = len(map)
    width = len(map[0])

    #stores the row and column counter
    row_counter = 0
    col_counter = 0

    #checks if the width is less than the height
    if(width < height):
        
        #loops through the 2d array to check for empty spaces
        for x in range(width):
            for y in range(height):
                #if the current column has empty space add to counter and skip to next column
                if map[y][x] == "":
                    col_counter += 1
                    break

        
        return col_counter

    #checks if the height is less than the width
    elif(height <= width):
        for y in range(height):
            for x in range(width):
                #if the current row has empty space add to counter and skip to next row
                if map[y][x] == "":
                    row_counter += 1
                    break
        
        return row_counter


        


    




def beam_search(problem, f, beam_width): 
    # Task 4
    # Implement a beam-width version A* search.
    # Return a search node containing a solved state.
    # Experiment with the beam width in the test code to find a solution.
    # Replace the line below with your code.
    display = False


    
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)

    

    frontier.append(node)

    while len(frontier.heap) > 50:
        frontier.pop()  # Removes the highest priority element (min or max based on order)
    






    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None

if __name__ == "__main__":

    # Task 1 test code
    
    print('The loaded initial state is visualised below.')
    visualise(read_initial_state_from_file('C:/Users/jaras/OneDrive - The University of Waikato/Programming/COMPX216/aima-python/aima-python/assignment1config.txt'))
    

    # Task 2 test code
    
    garden = ZenPuzzleGarden('C:/Users/jaras/OneDrive - The University of Waikato/Programming/COMPX216/aima-python/aima-python/assignment1config.txt')
    print('Running breadth-first graph search.')
    before_time = time()
    node = breadth_first_graph_search(garden)
    after_time = time()
    print(f'Breadth-first graph search took {after_time - before_time} seconds.')
    if node:
        print(f'Its solution with a cost of {node.path_cost} is animated below.')
        animate(node)
    else:
        print('No solution was found.')
    

    # Task 3 test code
    
    print('Running A* search.')
    before_time = time()
    node = astar_search(garden, astar_heuristic_cost)
    after_time = time()
    print(f'A* search took {after_time - before_time} seconds.')
    if node:
        print(f'Its solution with a cost of {node.path_cost} is animated below.')
        animate(node)
    else:
        print('No solution was found.')
    

    # Task 4 test code
    
    print('Running beam search.')
    before_time = time()
    node = beam_search(garden, lambda n: n.path_cost + astar_heuristic_cost(n), 50)
    after_time = time()
    print(f'Beam search took {after_time - before_time} seconds.')
    if node:
        print(f'Its solution with a cost of {node.path_cost} is animated below.')
        animate(node)
    else:
        print('No solution was found.')
    
