from util import Stack, Queue, PriorityQueue
from abc import ABC, abstractmethod
from typing import Iterator, Container, Callable
from game import GameState, Action


class SearchProblem(ABC):

    def __init__(self, gameState: GameState):
        self.startState = gameState

    @abstractmethod
    def getStartState(self):
        return NotImplementedError

    @abstractmethod
    def isGoalState(self, state):
        raise NotImplementedError

    @abstractmethod
    def getSuccessors(self, state):
        raise NotImplementedError

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
        return super().__call__(s, *args)


def nullHeuristic(state, problem=None):
    return 0


def searchIterator(problem: SearchProblem, history: Container[tuple[GameState, list[Action], float]], check: Callable, depth: int = 99999) -> Iterator[list[Action]]:
    """
    This is a general search function that takes a problem, a history data structure
    and a checker and return a list of actions
    """
    history.push(([problem.getStartState()], [], 0, 0))
    check(problem.getStartState(), [])
    while not history.isEmpty():
        S, A, C, d = history.pop()
        if problem.isGoalState(S[-1]):
            yield A
        for s, a, c in problem.getSuccessors(S[-1]):
            if d < depth and check(s, S):
                history.push((S + [s], A + [a], C + c, d + 1))


def search(problem: SearchProblem, history: Container[tuple[GameState, list[Action], float]], check: Callable, depth: int = 99999) -> list[Action]:
    iterator = searchIterator(problem, history, check, depth)
    try:
        return next(iterator)
    except StopIteration:
        return []


def searchAll(problem: SearchProblem, history: Container[tuple[GameState, list[Action], float]], check: Callable, depth: int = 99999) -> list[list[Action]]:
    return list(searchIterator(problem, history, check, depth))


def depthFirstSearch(problem: SearchProblem, depth: int = 99999):
    return search(problem, Stack(), cycleChecker(), depth)


def depthFirstSearchIterator(problem: SearchProblem, depth: int = 99999):
    return searchIterator(problem, Stack(), cycleChecker(), depth)


def breadthFirstSearch(problem: SearchProblem, depth: int = 99999):
    return search(problem, Queue(), visitChecker(), depth)


def breadthFirstSearchIterator(problem: SearchProblem, depth: int = 99999):
    return searchIterator(problem, Queue(), visitChecker(), depth)


def uniformCostSearch(problem: SearchProblem, depth: int = 99999):
    return search(problem, PriorityQueueForUCS(), goalExcludedVisitChecker(problem), depth)


def uniformCostSearchIterator(problem: SearchProblem, depth: int = 99999):
    return searchIterator(problem, PriorityQueueForUCS(), goalExcludedVisitChecker(problem), depth)


def aStarSearch(problem: SearchProblem, heuristic: Callable = nullHeuristic, depth: int = 99999):
    return search(problem, PriorityQueueForAStar(problem, heuristic), goalExcludedVisitChecker(problem), depth)


def aStarSearchIterator(problem: SearchProblem, heuristic: Callable = nullHeuristic, depth: int = 99999):
    return searchIterator(problem, PriorityQueueForAStar(problem, heuristic), goalExcludedVisitChecker(problem), depth)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
