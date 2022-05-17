import numpy as np

from agents import *
from tkinter import *
from search import *
import time

class myProblem(Problem):
    def actions(self, state):
        actions = ['NoOp', 'TurnRight', 'TurnLeft', 'Forward']
        location, direction, dirt_map = state
        xi, yi = location[0], location[1]

        if dirt_map[xi, yi]:
            actions.append('Suck')
        return actions

    def result(self, state, action):
        #TODO: check that action is in set of actions fiven by self.action --> if action in self.actions() ;
        # For this method be aware of the mechanics of the environment: Forward, e.g how will change the state, etc.

        location, direction, dirt_map = state
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

            state_update = (location_update, direction, dirt_map)
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

            state_update = (location, direction_update, dirt_map)
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

            state_update = (location, direction_update, dirt_map)
            return state_update

        elif action == 'Suck':
            dirt_map_update = dirt_map.copy()
            dirt_map_update[xi, yi] = False
            state_update = (location, direction, dirt_map_update)
            return state_update

        elif action == 'NoOp':
            state_update = (location, direction, dirt_map)
            return state_update

    def goal_test(self, state):
        location, direction, dirt_map = state
        return dirt_map.any() == self.goal.any() #TODO: any() or all()

    def h(self, node):
        def manhattanDistance(xy1, xy2):
            "Returns the Manhattan distance between points xy1 and xy2"
            return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

        location, direction, dirt_map = node.state
        food_points = np.where(dirt_map == True)

        #TODO: Correct if below
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




class myEnv(VacuumEnvironment):
    xi, yi = (0, 0)
    perceptible_distance = 1

    def __init__(self, root, width=12, height=12, elements=None):
        super().__init__(width, height)
        if elements is None:
            elements = ['D', 'W']
        self.width = width
        self.root = root
        self.create_frames()
        self.create_buttons()
        self.create_walls()
        self.elements = elements


    def create_frames(self):
        """Adds frames to the GUI environment."""
        self.frames = []
        for _ in range(self.width):
            frame = Frame(self.root, bg='grey')
            frame.pack(side='bottom')
            self.frames.append(frame)

    def create_buttons(self):
        """Adds buttons to the respective frames in the GUI."""
        self.buttons = []
        for frame in self.frames:
            button_row = []
            for _ in range(self.width):
                button = Button(frame, height=3, width=5, padx=2, pady=2)
                button.config(
                    command=lambda btn=button: self.display_element(btn))
                button.pack(side='left')
                button_row.append(button)
            self.buttons.append(button_row)

    def create_walls(self):
        """Creates the outer boundary walls which do not move."""
        for row, button_row in enumerate(self.buttons):
            if row == 0 or row == len(self.buttons) - 1:
                for button in button_row:
                    button.config(text='W', state='disabled',
                                  disabledforeground='black')
            else:
                button_row[0].config(
                    text='W', state='disabled', disabledforeground='black')
                button_row[len(button_row) - 1].config(text='W',
                                                       state='disabled', disabledforeground='black')
        # Place the agent in the centre of the grid.
        self.buttons[1][1].config(
            text='A', state='disabled', disabledforeground='black')

    def execute_action(self, agent, action):
        """Determines the action the agent performs."""
        xi, yi = (self.xi, self.yi)
        if action == 'Suck':
            dirt_list = self.list_things_at(agent.location, Dirt)
            if dirt_list:
                dirt = dirt_list[0]
                agent.performance += 100
                self.delete_thing(dirt)
                self.buttons[xi][yi].config(text='', state='normal')
                xf, yf = agent.location
                self.buttons[xf][yf].config(
                    text='A', state='disabled', disabledforeground='black')
                print("Vaccum sucked at position: ({}, {})".format(agent.location[0], agent.location[1]))
        else:
            agent.bump = False
            if action == 'TurnRight':
                agent.direction += Direction.R
            elif action == 'TurnLeft':
                agent.direction += Direction.L
            elif action == 'Forward':
                agent.bump = self.move_to(agent, agent.direction.move_forward(agent.location))

                if not agent.bump:
                    self.buttons[xi][yi].config(text='', state='normal')
                    xf, yf = agent.location
                    self.buttons[xf][yf].config(
                        text='A', state='disabled', disabledforeground='black')
                else:
                    print('Collision!')

        if action != 'NoOp':
            agent.performance -= 1

    def __get_dirt_map(self):
        state = np.full([self.width, self.height], False)
        for thing in self.things:
            if isinstance(thing, Dirt):
                state[thing.location[0]][thing.location[1]] = True
            #elif isinstance(thing, Wall):
            #    state[thing.location[0]][thing.location[1]] = -1
            #elif isinstance(thing, XYReflexAgent):
            #    state[thing.location[0]][thing.location[1]] = 10

        return state

    def percept(self, agent):
        dirt_map = self.__get_dirt_map()
        location = agent.location
        direction = agent.direction.direction
        state = (location, direction, dirt_map)
        return state

    def read_env(self):
        """Reads the current state of the GUI environment."""
        for i, btn_row in enumerate(self.buttons):
            for j, btn in enumerate(btn_row):
                if (i != 0 and i != len(self.buttons) - 1) and (j != 0 and j != len(btn_row) - 1):
                    agt_loc = self.agents[0].location
                    if self.some_things_at((i, j)) and (i, j) != agt_loc:
                        for thing in self.list_things_at((i, j)):
                            self.delete_thing(thing)
                    if btn['text'] == self.elements[0]:
                        self.add_thing(Dirt(), (i, j))
                    elif btn['text'] == self.elements[1]:
                        self.add_thing(Wall(), (i, j))

    def update_env(self):
        """Updates the GUI environment according to the current state."""
        self.read_env()
        agt = self.agents[0]
        previous_agent_location = agt.location
        self.xi, self.yi = previous_agent_location
        self.step()
        xf, yf = agt.location

    def run_env(self):
        n_steps = 10000
        isDone = False

        while not isDone:
            self.update_env()
            # env.run()
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.2)
            n_steps -= 1
            if n_steps == 0 or not self.percept(self.agents[0])[2].any():
                isDone = True
                print("Max number of steps completed")
                print("Agents performance: {}".format(self.agents[0].performance))
                exit()

    def display_element(self, button):
        """Show the things on the GUI."""
        txt = button['text']
        if txt != 'A':
            if txt == 'W':
                button.config(text='D')
            elif txt == 'D':
                button.config(text='')
            elif txt == '':
                button.config(text='W')


def XYReflexAgentProgram(percept):
    """The modified SimpleReflexAgentProgram for the GUI environment."""
    status, bump = percept

    """
    if status == 'Dirty':
        return 'Suck'

    if bump == 'Bump':
        print("There is a BUMP!")
        value = random.choice((1, 2))
    else:
        value = random.choice((1, 2, 3, 4))  # 1-right, 2-left, others-forward

    if value == 1:
        return 'TurnRight'
    elif value == 2:
        return 'TurnLeft'
    else:
        return 'Forward'
    """
    return 'Forward'


class XYReflexAgent(Agent):
    """The modified SimpleReflexAgent for the GUI environment."""
    def __init__(self, program=None):
        super().__init__(program)
        self.location = (1, 1)
        self.direction = Direction("right")
#######################################################################################
# Uninformed algorithm

def breadth_first_graph_search(problem):
    def is_same_state(state1, state2):
        if state1[0] != state2[0] or state1[1] != state2[1] or not (state1[2] == state2[2]).all():
            return False
        else:
            return True

    def not_in_explored(state, explored):
        counter = 0
        for exp_state in explored:
            if exp_state[0] != state[0] or exp_state[1] != state[1] or not (exp_state[2] == state[2]).all():
                counter += 1
        if counter == len(explored):
            return True
        else:
            return False

    def not_in_frontier(child, frontier):
        if len(frontier) == 0:
            return True
        else:
            counter = 0
            for fron in frontier:
                if child.action != fron.action or child.depth != fron.depth or child.parent != fron.parent or \
                        child.path_cost != fron.path_cost or not is_same_state(child.state, fron.state):
                    counter += 1
            if counter == len(frontier):
                return True
            else:
                return False


    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    # explored = set()
    explored = list()

    while frontier:
        node = frontier.popleft()
        #if node.state[0] == (1,2):
        #    print("Stop")
        # explored.add(node.state)

        #if node.state not in explored:
        if not_in_explored(node.state, explored):
            explored.append(node.state)

        for child in node.expand(problem):
            #if child.action == 'Suck':
            #    print("STOp")
            #if child not in frontier:
            if not_in_frontier(child, frontier):
                counter = 0
                if not_in_explored(child.state, explored):
                    if problem.goal_test(child.state):
                        print(len(explored), "paths have been expanded and", len(frontier),
                              "paths remain in the frontier")
                        return child.solution()
                    frontier.append(child)

    return None

##################################################################################################
# Informed algorithm

def best_first_graph_search(problem, f):
    def is_same_state(state1, state2):
        if state1[0] != state2[0] or state1[1] != state2[1] or not (state1[2] == state2[2]).all():
            return False
        else:
            return True

    def not_in_explored(state, explored):
        counter = 0
        for exp_state in explored:
            if exp_state[0] != state[0] or exp_state[1] != state[1] or not (exp_state[2] == state[2]).all():
                counter += 1
        if counter == len(explored):
            return True
        else:
            return False

    def not_in_frontier(child, frontier):
        if len(frontier) == 0:
            return True
        else:
            counter = 0
            for fron in frontier.heap:
                if child.action != fron[1].action or child.depth != fron[1].depth or child.parent != fron[1].parent or \
                        child.path_cost != fron[1].path_cost or not is_same_state(child.state, fron[1].state):
                    counter += 1
            if counter == len(frontier):
                return True
            else:
                return False

    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = list()
    #explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):

            print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        #explored.add(node.state)
        explored.append(node.state)
        for child in node.expand(problem):
            #if child not in frontier:
            if not_in_frontier(child, frontier) and not_in_explored(child.state, explored):
                    frontier.append(child)

            elif not not_in_frontier(child, frontier):
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
            """
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
            """
    return None

def astar_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


class vacuumAgentProgram(SimpleProblemSolvingAgentProgram):
    #def __init__(self, percept):
    #    super().__init__()
    #    super().__call__(percept)
    #    self.direction = Direction("up")

    def update_state(self, state, percept): #TODO: Check this method!!!!
        return percept

    def formulate_goal(self, state):
        location, direction, dirt_map = state
        goal = np.full(dirt_map.shape, False)

        return goal

    def formulate_problem(self, state, goal):
        #problem = state
        #return problem
        problem = myProblem(initial=state, goal=goal)
        return problem

    def search(self, problem):
        start_time = time.time()
        seq = breadth_first_tree_search(problem)
        #return breadth_first_graph_search(problem)
        #seq = astar_search(problem)

        end_time = time.time()
        elapsed = round(end_time - start_time, 2)
        print("Elapsed time: {}".format(elapsed))

        seq = seq.solution()
        return seq


def main():
    root = Tk()
    root.title("Vacuum Environment")
    root.geometry("740x660")
    root.resizable(0, 0)
    frame = Frame(root, bg='black')

    frame.pack(side='bottom')
    run_button = Button(frame, text='Run', height=2,
                         width=6, padx=2, pady=2)
    run_button.pack(side='left')
    frame.pack(side='bottom')
    env = myEnv(root, width=10, height=10)
    #agt = XYReflexAgent(program=XYReflexAgentProgram)
    #init_state = ...
    agt_program = vacuumAgentProgram()
    agt = XYReflexAgent(program=agt_program)

    env.add_thing(agt, location=(1, 1))
    run_button.config(command=env.run_env)

    root.mainloop()


if __name__ == '__main__':
    main()
