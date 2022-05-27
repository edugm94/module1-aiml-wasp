# AI-ML: Module 1 [WASP Course]

This repository contains the code of the assignment proposed in the WASP course AI and ML. 
This project proposes a solution for a intelligent vacuum cleaner agent by using search algorithms. 

The code is structured in different parts:
* myAgent.py: Definition of our vacuum cleaner agent. Statement and codification of the problem intended to solve.
* myEnv.py: Adaptation of the graphical environment for using it in this project. The function: "percept" plays an 
important role, and is adapted in this file.

Three different search algorithms are tested in this project: DFS (Deep-first search); BFS (Breadth-first search); and 
A* (informative search algorithm). To execute each of these algorithms, one must modify the `search` function by uncommenting
the desired algorithm. Such a function is found in the script ``myAgent.py`` under the class: `vacuumAgentProgram`.

To run the project, execute the following line of code:
```
python main.py
```
The graphical environment appears, giving the option to the user to place the obstacles and dirt spaces within the grid.
Once all objects are place, the search will start after clicking on ``run`` buttom.
