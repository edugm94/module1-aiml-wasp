import numpy as np
from utils import memoize, PriorityQueue
from search import Node
from collections import deque


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

#######################################################################################
# Uninformed algorithm


def depth_first_graph_search(problem):
    frontier = [(Node(problem.initial))]
    explored = list()

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            print(len(explored), "paths have been expanded and", len(frontier),
                  "paths remain in the frontier")
            return node
        explored.append(node.state)

        for child in node.expand(problem):
            if not_in_explored(child.state, explored) and not_in_frontier(child, frontier):
                #frontier.extend(child)
                frontier.append(child)
    return None


def breadth_first_graph_search(problem):
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
                        return child
                    frontier.append(child)

    return None

##################################################################################################
# Informed algorithm


def best_first_graph_search(problem, f):
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
    return None


def astar_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))
