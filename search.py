from util import Stack, Queue, PriorityQueue
from abc import ABC, abstractmethod
from typing import Iterator

from game import GameState, Action


class SearchProblem(ABC):

    def __init__(self, gameState: GameState):
        self.startingGameState = gameState

    @abstractmethod
    def getStartState(self):
        raise NotImplementedError

    @abstractmethod
    def isGoalState(self, state):
        raise NotImplementedError

    @abstractmethod
    def getSuccessors(self, state):
        raise NotImplementedError

    @abstractmethod
    def getCostOfActions(self, actions: list[Action]):
        raise NotImplementedError


class PriorityQueueForUCS(PriorityQueue):
    def push(self, item):
        super().push(item, item[2])


class PriorityQueueForAStar(PriorityQueue):
    def __init__(self, problem, heuristics):
        super().__init__()
        self.problem = problem
        self.heuristics = heuristics

    def push(self, item):
        super().push(item, item[2] +
                     self.heuristics(item[0][-1], self.problem))


class cycleChecker:
    """
    This is a checker that only checks whether the new position causes a cycle
    but do not check whether the new position is checked again (see visitCheck)
    """

    def __call__(self, pos, path) -> bool:
        return pos not in path


class visitChecker(set):
    """
    This is a checker that checks whether the state is visited
    each call of this checker will be recorded in set
    """

    def __call__(self, s, *_) -> bool:
        if s in self:
            return False
        self.add(s)
        return True


class goalExcludedVisitChecker(visitChecker):
    """
    This is a visitChecker which escape the case about checking goal
    """

    def __init__(self, problem):
        self.problem = problem

    def __call__(self, s, *args) -> bool:
        if self.problem.isGoalState(s):
            return True
        return super()(s, *args)


def nullHeuristic(state, problem=None):
    return 0


def searchIterator(problem: SearchProblem, history: list[tuple[GameState, list[Action], float]], check: callable) -> Iterator[list[Action]]:
    """
    This is a general search function that takes a problem, a history data structure
    and a checker and return a list of actions
    """
    history.push(([problem.getStartState()], [], 0))
    check(problem.getStartState(), [])
    while not history.isEmpty():
        S, A, C = history.pop()
        if problem.isGoalState(S[-1]):
            yield A
        for s, a, c in problem.getSuccessors(S[-1]):
            if check(s, S):
                history.push((S + [s], A + [a], C + c))


def search(problem: SearchProblem, history: list[tuple[GameState, list[Action], float]], check: callable) -> list[Action]:
    iterator = searchIterator(problem, history, check)
    try:
        return next(iterator)
    except StopIteration:
        return []
    
def searchAll(problem: SearchProblem, history: list[tuple[GameState, list[Action], float]], check: callable) -> list[list[Action]]:
    return list(searchIterator(problem, history, check))


def depthFirstSearch(problem: SearchProblem):
    return search(problem, Stack(), cycleChecker())

def depthFirstSearchIterator(problem: SearchProblem):
    return searchIterator(problem, Stack(), cycleChecker())

def breadthFirstSearch(problem: SearchProblem):
    return search(problem, Queue(), visitChecker())

def breadthFirstSearchIterator(problem: SearchProblem):
    return searchIterator(problem, Queue(), visitChecker())

def uniformCostSearch(problem: SearchProblem):
    return search(problem, PriorityQueueForUCS(), goalExcludedVisitChecker(problem))

def uniformCostSearchIterator(problem: SearchProblem):
    return searchIterator(problem, PriorityQueueForUCS(), goalExcludedVisitChecker(problem))

def aStarSearch(problem: SearchProblem, heuristic: callable = nullHeuristic):
    return search(problem, PriorityQueueForAStar(problem, heuristic), goalExcludedVisitChecker(problem))

def aStarSearchIterator(problem: SearchProblem, heuristic: callable = nullHeuristic):
    return searchIterator(problem, PriorityQueueForAStar(problem, heuristic), goalExcludedVisitChecker(problem))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
