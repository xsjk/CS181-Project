from abc import ABCMeta
from enum import Enum
import time
import signal
import random
import inspect
import sys
import heapq
import math
from typing import Any, Type, TypeVar, Iterable, Callable, Union, overload, List
from dataclasses import dataclass, field
from functools import partial, wraps
import types
import threading
import math

T = TypeVar('T')

def convert_arg(arg, target_type, verbose: bool = False):
    if verbose:
        print(f'Converting {arg} to {target_type.__name__ if type(target_type) == type else target_type}')
    if type(target_type) == str:
        return convert_arg(arg, eval(target_type))
    elif type(target_type) == type:
        if target_type == inspect.Parameter.empty or isinstance(arg, target_type):
            return arg
        try:
            return target_type(arg)
        except (ValueError, TypeError):
            return arg
    elif type(target_type) == types.GenericAlias:
        outer_type = target_type.__origin__
        inner_types = target_type.__args__
        if outer_type in (list, tuple, set):
            return outer_type(convert_arg(item, inner_types[0]) for item in arg)
        elif outer_type in (dict,):
            return outer_type((convert_arg(key, inner_types[0]), convert_arg(value, inner_types[1])) for key, value in arg.items())
        else:
            raise TypeError(f'Unknown type {target_type}')
    elif type(target_type) == types.UnionType:
        target_types = target_type.__args__
        # TODO
        return arg
    elif type(target_type) == types.NoneType:
        return None
    else:
        raise TypeError(f'Unknown type {target_type}')
    
def assert_arg(arg, target_type, verbose: bool = False):
    if verbose:
        print(f'Asserting {arg} to {target_type}')
    if type(target_type) == str:
        assert type(arg).__name__ == target_type, TypeError(f'Argument should be "{target_type}" but got "{type(arg).__name__}"')
    elif type(target_type) == type:
        if target_type == inspect.Parameter.empty or isinstance(arg, target_type):
            return arg
        raise TypeError(f'Argument should be "{target_type.__name__}" but got "{type(arg).__name__}"')
    elif type(target_type) == types.GenericAlias:
        outer_type = target_type.__origin__
        inner_types = target_type.__args__
        if outer_type in (list, tuple, set):
            return outer_type(assert_arg(item, inner_types[0]) for item in arg)
        elif outer_type in (dict,):
            return outer_type((assert_arg(key, inner_types[0]), assert_arg(value, inner_types[1])) for key, value in arg.items())
        else:
            raise TypeError(f'Unknown type {target_type}')
    elif type(target_type) == types.UnionType:
        pass
    # enum
    elif issubclass(target_type, Enum):
        if isinstance(arg, target_type):
            return arg
        elif isinstance(arg, str):
            return target_type[arg]
        else:
            raise TypeError(f'Argument should be "{target_type.__name__}" or "{target_type.__name__}" but got "{type(arg).__name__}"')
    else:
        raise TypeError(f'Unknown type {target_type}')
    

def auto_convert(verbose: bool = False):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            signature = inspect.signature(func)
            parameters = signature.parameters
            converted_args = [
                arg if (i, param.name)==(0,"self") else \
                      convert_arg(arg, param.annotation, verbose)
                for i, (arg, param) in enumerate(zip(args, parameters.values()))
                
            ]
            converted_kwargs = {
                key: convert_arg(value, parameters[key].annotation, verbose)
                for key, value in kwargs.items()
            }
            res = func(*converted_args, **converted_kwargs)
            return convert_arg(res, signature.return_annotation, verbose)
        return wrapper
    if callable(verbose):
        return decorator(verbose)
    return decorator
    
def type_check(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        signature = inspect.signature(func)
        parameters = signature.parameters
        for arg, param in zip(args, parameters.values()):
            assert_arg(arg, param.annotation)
        for key, value in kwargs.items():
            assert_arg(value, parameters[key].annotation)
        res = func(*args, **kwargs)
        assert_arg(res, signature.return_annotation)
        return res
    return wrapper

class classproperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()
    
class Singleton(ABCMeta):
    """ 
    This is a metaclass for classes that should only have one instance.
    If a class is instantiated with the twice, the same instance is returned.

    Usage:
    class MyClass(metaclass=Singleton):
        pass
    """
    _instances: dict[type, Any] = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
    @classmethod
    def __getitem__(cls, key):
        return cls._instances[key]

class Uniqueton(Singleton):
    """
    This is a metaclass for classes that should only have one instance.
    If a class is instantiated with the twice, an error is raised.

    Usage:
    class MyClass(metaclass=Uniqueton):
        pass
    """
    def __call__(cls, *args, **kwargs):
        if cls in cls._instances:
            raise RuntimeError(f'{cls.__name__} is already instantiated')
        return super().__call__(*args, **kwargs)


class Size:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.length = min(width, height)
        self.sizeTuple = (self.width, self.height)

class Vector2d:

    @overload
    def __init__(self, x: float, y: float):
        pass
    
    @overload
    def __init__(self, vec: Union[tuple[float, float], "Vector2d"]):
        pass

    def __init__(self, *args):
        match len(args):
            case 1:
                if isinstance(args[0], tuple):
                    self.x, self.y = args[0]
                elif isinstance(args[0], Vector2d):
                    self.x, self.y = args[0].x, args[0].y
                else:
                    raise TypeError
            case 2:
                self.x, self.y = args
            case _:
                raise TypeError

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r}, {!r})'.format(class_name, *self)

    def __str__(self):
        return str(tuple(self))

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def angle(self):
        return math.atan2(self.y, self.x)

    def __bool__(self):
        return bool(abs(self))

    def __format__(self, format_spec = ''):
        if format_spec.endswith('p'):
            format_spec = format_spec[:-1]
            coords = (abs(self), self.angle())
            outer_format = '<{}, {}>'
        else:
            coords = self
            outer_format = '({}, {})'
        components = (format(c, format_spec) for c in coords)
        return outer_format.format(*components)
    
    def __hash__(self):
        return super().__hash__()
    
    def __add__(self, other: "Vector2d") -> "Vector2d":
        return Vector2d(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: "Vector2d") -> "Vector2d":
        return Vector2d(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> "Vector2d":
        return Vector2d(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar: float) -> "Vector2d":
        return self * scalar
    
    def __neg__(self) -> "Vector2d":
        return Vector2d(-self.x, -self.y)
    
    def __iter__(self):
        return iter((self.x, self.y))
    
    def __getitem__(self, index: int) -> float:
        match index:
            case 0:
                return self.x
            case 1:
                return self.y
            case _:
                raise IndexError

    def __setitem__(self, index: int, value: float):
        match index:
            case 0:
                self.x = value
            case 1:
                self.y = value
            case _:
                raise IndexError

    
    def map(self, func):
        return Vector2d(func(self.x), func(self.y))
    
    @staticmethod
    def manhattanDistance(v1: "Vector2d", v2: "Vector2d") -> float:
        return abs(v1.x - v2.x) + abs(v1.y - v2.y)
    
    @staticmethod
    def euclideanDistance(v1: "Vector2d", v2: "Vector2d") -> float:
        return abs(v1 - v2)
    
V = Vector2d

class DisjointSet(list):
    @dataclass
    class Node:
        parent: int
        value: Any = None
        rank: field(init=False) = 0

    def __init__(self, n=None):
        if n != None:
            self.extend([self.Node(i, v) for i, v in enumerate(n)])
            self.lookup = {node.value: node.parent for node in self}

    def _find(self, i: int) -> int:
        if self[i].parent != i:
            self[i].parent = self._find(self[i].parent)
        return self[i].parent

    def merge(self, i: Any, j: Any):
        i, j = self.lookup[i], self.lookup[j]
        i, j = self._find(i), self._find(j)
        if i == j:
            pass
        elif self[i].rank < self[j].rank:
            self[i].parent = j
        elif self[i].rank > self[j].rank:
            self[j].rank = i
        else:
            self[i].parent = j
            self[j].rank += 1


class VectorBool:
    data: int = 0

    def __init__(self, size: int = None, data: int = None):
        if size != None:
            self.data = ((1 << size)-1)
        elif data != None:
            self.data = data

    def __getitem__(self, index: int) -> bool:
        return bool(self.data & (1 << index))

    def __setitem__(self, index: int, value: bool) -> None:
        self.data = self.data & ~(1 << index) | (value << index)

    def __int__(self) -> int:
        return self.data

    def __repr__(self) -> str:
        return bin(self.data[-3::-1])

    def __str__(self) -> str:
        return bin(self.data[-3::-1])

    def __eq__(self, __o) -> bool:
        return __o.data == self.data

    def __hash__(self) -> int:
        return hash(self.data)

    def copy(self):
        return VectorBool(data=self.data)


class Stack:
    "A container with a last-in-first-out (LIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Push 'item' onto the stack"
        self.list.append(item)

    def pop(self):
        "Pop the most recently pushed item from the stack"
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the stack is empty"
        return len(self.list) == 0


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0


class PriorityQueue(Queue):
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class PriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """

    def __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction      # store the priority function
        PriorityQueue.__init__(self)        # super-class initializer

    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(item))



"""
  Data structures and functions useful for various course projects

  The search project should not need anything below this line.
"""


class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print(a['test'])

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print(a['test'])
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print(a['test'])
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print(a['blah'])
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()
        def compare(x, y): return sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0:
            return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__(self, y):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__(self, y):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend


def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" %
          (method, line, fileName))
    sys.exit(1)


def normalize(vectorOrCounter):
    """
    normalize a vector or counter by dividing each value by the sum of all values
    """
    normalizedCounter = Counter()
    if type(vectorOrCounter) == type(normalizedCounter):
        counter = vectorOrCounter
        total = float(counter.totalCount())
        if total == 0:
            return counter
        for key in counter.keys():
            value = counter[key]
            normalizedCounter[key] = value / total
        return normalizedCounter
    else:
        vector = vectorOrCounter
        s = float(sum(vector))
        if s == 0:
            return vector
        return [el / s for el in vector]


def nSample(distribution, values, n):
    if sum(distribution) != 1:
        distribution = normalize(distribution)
    rand = [random.random() for i in range(n)]
    rand.sort()
    samples = []
    samplePos, distPos, cdf = 0, 0, distribution[0]
    while samplePos < n:
        if rand[samplePos] < cdf:
            samplePos += 1
            samples.append(values[distPos])
        else:
            distPos += 1
            cdf += distribution[distPos]
    return samples


def sample(distribution, values=None):
    if type(distribution) == Counter:
        items = sorted(distribution.items())
        distribution = [i[1] for i in items]
        values = [i[0] for i in items]
    if sum(distribution) != 1:
        distribution = normalize(distribution)
    choice = random.random()
    i, total = 0, distribution[0]
    while choice > total:
        i += 1
        total += distribution[i]
    return values[i]


def sampleFromCounter(ctr):
    items = sorted(ctr.items())
    return sample([v for k, v in items], [k for k, v in items])


def getProbability(value, distribution, values):
    """
      Gives the probability of a value under a discrete distribution
      defined by (distributions, values).
    """
    total = 0.0
    for prob, val in zip(distribution, values):
        if val == value:
            total += prob
    return total


def flipCoin(p):
    r = random.random()
    return r < p


def chooseFromDistribution(distribution):
    "Takes either a counter or a list of (prob, key) pairs and samples"
    if type(distribution) == dict or type(distribution) == Counter:
        return sample(distribution)
    r = random.random()
    base = 0.0
    for prob, element in distribution:
        base += prob
        if r <= base:
            return element


def nearestPoint(pos):
    """
    Finds the nearest grid point to a position (discretizes).
    """
    (current_row, current_col) = pos

    grid_row = int(current_row + 0.5)
    grid_col = int(current_col + 0.5)
    return (grid_row, grid_col)


def sign(x):
    """
    Returns 1 or -1 depending on the sign of x
    """
    if (x > 0):
        return 1
    elif(x == 0):
        return 0
    else:
        return -1


def isOdd(x: int) -> bool:
    return bool(x % 2)

def arrayInvert(array):
    """
    Inverts a matrix stored as a list of lists.
    """
    result = [[] for i in array]
    for outer in array:
        for inner in range(len(outer)):
            result[inner].append(outer[inner])
    return result


def matrixAslist(matrix, value=True):
    """
    Turns a matrix into a list of coordinates matching the specified value
    """
    rows, cols = len(matrix), len(matrix[0])
    cells = []
    for row in range(rows):
        for col in range(cols):
            if matrix[row][col] == value:
                cells.append((row, col))
    return cells


def pause():
    """
    Pauses the output stream awaiting user feedback.
    """
    input("<Press enter/return to continue>")


_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None
_MUTED = False


class WritableNull:
    def write(self, string):
        pass


def mutePrint():
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _MUTED
    if _MUTED:
        return
    _MUTED = True

    _ORIGINAL_STDOUT = sys.stdout
    # _ORIGINAL_STDERR = sys.stderr
    sys.stdout = WritableNull()
    # sys.stderr = WritableNull()


def unmutePrint():
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _MUTED
    if not _MUTED:
        return
    _MUTED = False

    sys.stdout = _ORIGINAL_STDOUT
    # sys.stderr = _ORIGINAL_STDERR



def addEach(x, y):
    if isinstance(x, Iterable) and isinstance(y, Iterable):
        return type(x)(map(addEach, x, y))
    else:
        return x + y


def subEach(x, y):
    if isinstance(x, Iterable) and isinstance(y, Iterable):
        return type(x)(map(subEach, x, y))
    else:
        return x - y


def mulEach(x, y):
    if isinstance(x, Iterable) and isinstance(y, Iterable):
        return type(x)(map(mulEach, x, y))
    else:
        return x * y


def deepmap(f: Callable, x: Any):
    if isinstance(x, Iterable):
        return type(x)(map(partial(deepmap, f), x))
    else:
        return f(x)
    