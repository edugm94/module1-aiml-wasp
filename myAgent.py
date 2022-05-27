from agents import Agent, Direction
from search import SimpleProblemSolvingAgentProgram, Problem
from algorithms import breadth_first_graph_search, depth_first_graph_search, astar_search
from search import depth_first_tree_search, breadth_first_tree_search
import time
import numpy as np


class VacuumAgent(Agent):
    """The modified SimpleReflexAgent for the GUI environment."""
    def __init__(self, program=None):
        super().__init__(program)
        self.location = (1, 1)
        self.direction = Direction("right")


class myProblem(Problem):
    def actions(self, state):
        actions = ['NoOp', 'TurnRight', 'TurnLeft']
        location, direction, dirt_map, walls = state
        xi, yi = location[0], location[1]

        if direction == 'up' and not walls[xi, yi - 1]:
            actions.append('Forward')
        elif direction == 'right' and not walls[xi + 1, yi]:
            actions.append('Forward')
        elif direction == 'down' and not walls[xi, yi + 1]:
            actions.append('Forward')
        elif direction == 'left' and not walls[xi - 1, yi ]:
            actions.append('Forward')

        if dirt_map[xi, yi]:
            actions.append('Suck')

        return actions

    def result(self, state, action):
        location, direction, dirt_map, walls = state
        xi, yi = location[0], location[1]

        # Variables to control bounds within the grid
        low_bound = 0
        high_bound = dirt_map.shape[0] - 1

        if action == 'Forward':
            if direction == 'up':
                xi_update = xi
                yi_update = yi - 1
                if yi_update == low_bound:
                    yi_update = yi
                location_update = (xi_update, yi_update)

            elif direction == 'right':
                xi_update = xi + 1
                yi_update = yi
                if xi_update == high_bound:
                    xi_update = xi
                location_update = (xi_update, yi_update)

            elif direction == 'down':
                xi_update = xi
                yi_update = yi + 1
                if yi_update == high_bound:
                    yi_update = yi
                location_update = (xi_update, yi_update)

            elif direction == 'left':
                xi_update = xi - 1
                yi_update = yi
                if xi_update == low_bound:
                    xi_update = xi
                location_update = (xi_update, yi_update)

            if walls[xi_update, yi_update] == True:
                xi_update = xi
                yi_update = yi
                location_update = (xi_update, yi_update)

            state_update = (location_update, direction, dirt_map, walls)
            return state_update

        elif action == 'TurnLeft':
            if direction == 'up':
                direction_update = 'left'
            elif direction == 'right':
                direction_update = 'up'
            elif direction == 'down':
                direction_update = 'right'
            elif direction == 'left':
                direction_update = 'down'

            state_update = (location, direction_update, dirt_map, walls)
            return state_update

        elif action == 'TurnRight':
            if direction == 'up':
                direction_update = 'right'
            elif direction == 'right':
                direction_update = 'down'
            elif direction == 'down':
                direction_update = 'left'
            elif direction == 'left':
                direction_update = 'up'

            state_update = (location, direction_update, dirt_map, walls)
            return state_update

        elif action == 'Suck':
            dirt_map_update = dirt_map.copy()
            dirt_map_update[xi, yi] = False
            state_update = (location, direction, dirt_map_update, walls)
            return state_update

        elif action == 'NoOp':
            state_update = (location, direction, dirt_map, walls)
            return state_update

    def goal_test(self, state):
        location, direction, dirt_map, walls = state
        return dirt_map.any() == self.goal.any()

    def h(self, node):

        def manhattanDistance(xy1, xy2):
            "Returns the Manhattan distance between points xy1 and xy2"
            return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

        location, direction, dirt_map, walls = node.state
        food_points = np.where(dirt_map == True)

        if food_points[0].size == 0:
            return 0
        else:
            min = np.inf

            for i in range(food_points[0].shape[0]):
                food_x = (food_points[0][i], food_points[1][i])
                distance = manhattanDistance(location, food_x)

                if min > distance:
                    min = distance
        return min


class vacuumAgentProgram(SimpleProblemSolvingAgentProgram):

    def update_state(self, state, percept): #TODO: Check this method!!!!
        return percept

    def formulate_goal(self, state):
        location, direction, dirt_map, walls = state
        goal = np.full(dirt_map.shape, False)

        return goal

    def formulate_problem(self, state, goal):
        problem = myProblem(initial=state, goal=goal)
        return problem

    def search(self, problem):
        start_time = time.time()
        #seq = depth_first_tree_search(problem)
        #seq = breadth_first_tree_search(problem)

        #seq = depth_first_graph_search(problem)
        #seq = breadth_first_graph_search(problem)
        seq = astar_search(problem) #Guess it will finish later, so we can text later,ch(problem)

        end_time = time.time()
        elapsed = round(end_time - start_time, 2)
        print("Elapsed time: {}".format(elapsed))

        seq = seq.solution()
        return seq