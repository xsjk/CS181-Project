from display import Display
import textual
from textual.app import App, CSSPathType, ComposeResult
from textual import events, on
from textual.driver import Driver
from util import Vector2d, Size

class TextualGraphics(Display):

    app: App

    def __init__(self, map_size: Vector2d, tile_size: Vector2d):
        self.MAP_SIZE = Size(map_size.x, map_size.y)
        self.TILE_SIZE = Size(tile_size.x, tile_size.y)
        self.WINDOW_SIZE = Size(self.MAP_SIZE.width * self.TILE_SIZE.width,
                                self.MAP_SIZE.height * self.TILE_SIZE.height)
        self.radius = self.TILE_SIZE.length * 0.8 / 2
        self._state = None
        self._frame = None
        self._frame = self.app.display.frame("main")
        self._frame.add_view(self, 0)
        self._frame.add_key_binding("q", "quit", "Quit")
        self._frame.add_key_binding("r", "refresh", "Refresh")
        self._frame.add_key_binding("p", "pause", "Pause")

    def initialize(self, state):
        self._state = state
        self._frame.refresh()

    def update(self, state):
        self._state = state
        self._frame.refresh()

    def draw(self, state):
        pass

class TextualKeyboardAgent(Display):
    pass