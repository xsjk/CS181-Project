import math

class Vector2d(tuple):
    def __new__(cls, x, y):
        return tuple.__new__(cls, (x, y))

    def __init__(self, x, y):
        self.x = x
        self.y = y

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
    
    def map(self, func):
        return Vector2d(func(self.x), func(self.y))
    
    @staticmethod
    def manhattanDistance(v1: "Vector2d", v2: "Vector2d") -> float:
        return abs(v1.x - v2.x) + abs(v1.y - v2.y)
    
    @staticmethod
    def euclideanDistance(v1: "Vector2d", v2: "Vector2d") -> float:
        return abs(v1 - v2)
    
    
V = Vector2d

if __name__ == "__main__":
    v1 = Vector2d(3, 4)
    print(v1.x, v1.y)
    x, y = v1
    print(x, y)
    print(v1)
    v1_clone = eval(repr(v1))
    print(v1 == v1_clone)
    print(v1)
    octets = bytes(v1)
    print(octets)
    print(abs(v1))
    print(bool(v1), bool(Vector2d(0, 0)))
    print(format(v1))
    print(format(v1, '.2f'))
    print(format(v1, '.3e'))
    print(format(Vector2d(1, 1), 'p'))
    print(format(Vector2d(1, 1), '.3ep'))
    print(format(Vector2d(1, 1), '0.5fp'))
    print(hash(v1))
    print(hash(v1_clone))
    print(len(set([v1, v1_clone])))
    print(v1.x, v1.y)
    print(v1[0], v1[1])
    try:
        v1.x = 7
    except AttributeError as e:
        print(e)
    try:
        v1[0] = 7
    except TypeError as e:
        print(e)
    try:
        v1_clone.x = 7
    except AttributeError as e:
        print(e)
    try:
        v1_clone[0] = 7
    except TypeError as e:
        print(e)
            
    v1 = Vector2d(3, 4)
    v2 = Vector2d(3.1, 4.2)
    print(v1 + v2)
    print(v1 * 2)
    print(2 * v1)
    print(abs(v1))
    print(abs(v1 * 2))
    print(bool(v1))