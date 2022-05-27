import numpy as np

from agents import VacuumEnvironment, Direction, Dirt, Wall
from tkinter import *
from search import *
import time


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

        return state

    def __get_walls_map(self):
        state = np.full([self.width, self.height], False)
        for thing in self.things:
            if isinstance(thing, Wall):
                state[thing.location[0]][thing.location[1]] = True

        return state

    def percept(self, agent):
        dirt_map = self.__get_dirt_map()
        location = agent.location
        direction = agent.direction.direction
        walls = self.__get_walls_map()
        state = (location, direction, dirt_map, walls)
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
        n_actions = 0
        isDone = False

        while not isDone:
            self.update_env()
            # env.run()
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.2)
            n_steps -= 1
            n_actions += 1
            if n_steps == 0 or not self.percept(self.agents[0])[2].any():
                isDone = True
                print("Room cleaned, task finished!")
                print("Agents performance: {}".format(self.agents[0].performance))
                print("Total number of actions: {}".format(n_actions))
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
