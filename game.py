from util import *
from agentRules import *
import traceback
from layout import Layout
from displayModes import GraphicMode


class PlayerRules:
    """
    These functions govern how player interacts with his environment under
    the classic game rules.
    """
    PLAYER_SPEED = 1

    @staticmethod
    def getLegalActions(state: "GameState"):
        """
        Returns a list of possible actions.d
        """
        return Actions.getPossibleActions(state.getPlayerState().configuration, state.layout)

    @staticmethod
    def applyAction(state: "GameState", action: Action):
        """
        Edits the state to reflect the results of the action.
        """
        assert isinstance(action, Action)
        
        # When action is TP
        if action == Action.TP:
            vector = Actions.translateVector(state)
            print("You use TP!")
        else:
            legal = PlayerRules.getLegalActions(state)
            # print("The legal actions are",legal)
            # print("The input action is", action)
            if action not in legal:
                raise Exception(f"Illegal action {action}")
            # Update Configuration
            vector = Actions.actionToVector(action, PlayerRules.PLAYER_SPEED)
        
        playerState = state.agentStates[0]

        playerState.configuration = playerState.configuration.getNextState(
            vector)


class GhostRules:
    """
    These functions dictate how ghosts interact with their environment.
    """
    GHOST_SPEED = 1.0

    @staticmethod
    def getLegalActions(state: "GameState", ghostIndex: int):
        """
        Ghosts cannot stop, and cannot turn around unless they
        reach a dead end, but can turn 90 degrees at intersections.
        """
        conf = state.getGhostState(ghostIndex).configuration
        possibleActions = Actions.getPossibleActions(conf, state.layout)
        # reverse = Actions.reverseDirection(conf.direction)
        # if reverse in possibleActions and len(possibleActions) > 1:
        #     possibleActions.remove(reverse)
        return possibleActions

    @staticmethod
    def applyAction(state: "GameState", action: Action, ghostIndex: int):
        assert isinstance(action, Action)
        legal = GhostRules.getLegalActions(state, ghostIndex)
        # print("The legal actions are:",legal)
        if action not in legal:
            raise Exception(f"Illegal ghost action {action}")

        ghostState = state.agentStates[ghostIndex]
        if ghostState.dead:
            return
        speed = GhostRules.GHOST_SPEED
        vector = Actions.actionToVector(action, speed)
        ghostState.configuration = ghostState.configuration.getNextState(
            vector)

    @staticmethod
    def checkDeath(state: "GameState", agentIndex: int):
        playerPosition = state.getPlayerPosition()
        if agentIndex == 0:  # Player just moved; Anyone can kill him
            for index in range(1, len(state.agentStates)):
                ghostState = state.agentStates[index]
                ghostPosition = ghostState.getPosition()
                if GhostRules.canKill(playerPosition, ghostPosition):
                    GhostRules.collide(state)
        else:
            ghostState = state.agentStates[agentIndex]
            ghostPosition = ghostState.getPosition()
            for i in range(1, len(state.agentStates)):
                if i != agentIndex and GhostRules.canKill(state.agentStates[i].getPosition(), state.agentStates[agentIndex].getPosition()):
                    state.agentStates[i].color = COLOR["explosion"]
                    state.agentStates[i].dead = True
                    state.agentStates[agentIndex].color = COLOR["explosion"]
                    state.agentStates[agentIndex].dead = True
                    # print("ghosts collides")
            if GhostRules.canKill(playerPosition, ghostPosition):
                GhostRules.collide(state)

    @staticmethod
    def canKill(playerPosition: Vector2d, ghostPosition: Vector2d) -> bool:
        return playerPosition == ghostPosition

    @staticmethod
    def collide(state):
        if not state._win:
            state.score -= 500
            state._lose = True

    @staticmethod
    def placeGhost(state: "GameState", ghostState: "AgentState"):
        ghostState.configuration = ghostState.start


class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def newGame(self, layout: Layout, playerAgent: Agent, ghostAgents: list[Agent], display: GraphicMode, quiet: bool = False, catchExceptions: bool = False):
        # print(ghostAgents)
        agents = [playerAgent] + ghostAgents
        initState = GameState()
        initState.initialize(layout)
        game = Game(agents, display, self, catchExceptions=catchExceptions)
        game.state = initState
        self.initialState = initState.deepCopy()
        self.display = display
        self.quiet = quiet
        return game

    def process(self, state: "GameState", game: "Game"):
        """
        Checks to see whether it is time to end the game.
        """
        if state.isWin():
            self.win(state, game)
        if state.isLose():
            self.lose(state, game)

    def win(self, state: "GameState", game: "Game"):
        if not self.quiet:
            print("Player emerges victorious! Score: %d" % state.data.score)
        game.gameOver = True

    def lose(self, state: "GameState", game: "Game"):
        if not self.quiet:
            print("Player died! Score: %d" % state.score)
        game.gameOver = True

    def agentCrash(self, game: "Game", agentIndex: int):
        if agentIndex == 0:
            print("Player crashed")
        else:
            print("A ghost crashed")


class GameState:

    def __init__(self, prevState=None):
        """
        Generates a new data packet by copying information from its predecessor.
        """
        self.agentStates: list[AgentState]

        self._agentMoved = None
        self._lose: bool = False
        self._win: bool = False
        self.scoreChange = 0
        self.score = 0
        self.layout: Layout

        if prevState != None:
            self.agentStates = self.copyAgentStates(prevState.agentStates)
            self.score = prevState.score
            self.layout = prevState.layout
            self.scoreChange = prevState.scoreChange

    def deepCopy(self):
        state = GameState(self)
        # state._agentMoved = self._agentMoved
        return state

    def copyAgentStates(self, agentStates: list[AgentState]):
        copiedStates = []
        for agentState in agentStates:
            copiedStates.append(agentState.copy())
        return copiedStates

    def __eq__(self, other: "GameState") -> bool:
        """
        Allows two states to be compared.
        """
        if other == None:
            return False
        # TODO Check for type of other
        if not self.agentStates == other.agentStates:
            return False
        if not self.score == other.score:
            return False
        return True

    def __hash__(self) -> int:
        """
        Allows states to be keys of dictionaries.
        """
        for i, state in enumerate(self.agentStates):
            try:
                int(hash(state))
            except TypeError as e:
                print(e)
                # hash(state)
        return int((hash(tuple(self.agentStates))))

    def initialize(self, layout: Layout):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        # self.capsules = []
        self.score = 0
        self.scoreChange = 1

        self.agentStates = []
        numGhosts = 0

        self.layout = layout

        pos = layout.agentPositions[0]
        self.agentStates.append(AgentState(
            Configuration(pos, Direction.NORTH.vector), True))
        for i in range(1, layout.ghostNum+1):
            pos = layout.agentPositions[i]
            self.agentStates.append(AgentState(
                Configuration(pos, Direction.NORTH.vector), False))

    # static variable keeps track of which states have had getLegalActions called
    explored = set()

    @staticmethod
    def getAndResetExplored():
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp

    def getLegalActions(self, agentIndex: int = 0):
        """
        Returns the legal actions for the agent specified.
        """
#        GameState.explored.add(self)
        if self.isWin() or self.isLose():
            return []

        if agentIndex == 0:  # Player is moving
            return PlayerRules.getLegalActions(self)
        else:
            return GhostRules.getLegalActions(self, agentIndex)

    @type_check
    def getNextState(self, agentIndex: int, action: Action) -> "GameState":
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.isWin() or self.isLose():
            raise Exception("Can't generate a successor of a terminal state.")

        # Copy current state
        state = GameState(self)

        # Let agent's logic deal with its action's effects on the board
        if agentIndex == 0:  # Player is moving
            state._eaten = [False for i in range(state.getNumAgents())]
            PlayerRules.applyAction(state, action)
            state.score += state.scoreChange
        else:                # A ghost is moving
            GhostRules.applyAction(state, action, agentIndex)

        # Resolve multi-agent effects
        GhostRules.checkDeath(state, agentIndex)

        # print(state.scoreChange)
        # Book keeping
        state._agentMoved = agentIndex
        GameState.explored.add(self)
        GameState.explored.add(state)

        return state

    def getLegalPlayerActions(self) -> list[Action]:
        return self.getLegalActions(0)

    def getPlayerNextState(self, action: Action) -> "GameState":
        """
        Generates the successor state after the specified player move
        """
        return self.getNextState(0, action)

    def getPlayerState(self) -> AgentState:
        """
        Returns an AgentState object for player (in game.py)

        state.pos gives the current position
        state.direction gives the travel vector
        """
        return self.agentStates[0].copy()
    
    def getAgentState(self, agentIndex: int) -> AgentState:
        return self.agentStates[agentIndex]

    def getPlayerPosition(self) -> Vector2d:
        return self.agentStates[0].getPosition()

    def getGhostStates(self):
        return self.agentStates[1:]

    def getGhostState(self, agentIndex: int) -> AgentState:
        if agentIndex == 0 or agentIndex >= self.getNumAgents():
            raise Exception("Invalid index passed to getGhostState")
        return self.agentStates[agentIndex]

    def getGhostPosition(self, agentIndex: int) -> Vector2d:
        if agentIndex == 0:
            raise Exception("Player's index passed to getGhostPosition")
        return self.agentStates[agentIndex].getPosition()

    def getGhostPositions(self) -> list[Vector2d]:
        return [s.getPosition() for s in self.getGhostStates()]

    def getNumAgents(self) -> int:
        return len(self.agentStates)

    def getMapSize(self) -> Vector2d:
        return self.layout.map_size

    def getScore(self) -> float:
        return float(self.score)

    def isLose(self) -> bool:
        return self._lose

    def isWin(self) -> bool:
        return self._win


# def gridToPixel(pos: tuple) -> Vector2d:
#     return (pos[0] * TILE_SIZE.width - TILE_SIZE.width // 2, pos[1] * TILE_SIZE.height - TILE_SIZE.height // 2)


def isOdd(x: int) -> bool:
    return bool(x % 2)

# The main controll flow


class Game:
    def __init__(self, agents: list[Agent], display, gameRule: ClassicGameRules, catchExceptions):
        # pygame.init()
        self.display = display

        self.agents = agents
        self.rules = gameRule

        self.startingIndex = 0
        self.gameOver = False
        self.catchExceptions = catchExceptions
        self.moveHistory = []
        self.score = 0
        self.state: GameState

    def _agentCrash(self, agentIndex, quiet=False):
        "Helper method for handling agent crashes"
        if not quiet:
            traceback.print_exc()
        self.gameOver = True
        self.agentCrashed = True
        self.rules.agentCrash(self, agentIndex)

    def run(self) -> None:
        self.numMoves = 0
        action = None

        # TODO: Update the action
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                print("Agent %d failed to load" % i, file=sys.stderr)
                self._agentCrash(i, quiet=True)
                return

        agentIndex = self.startingIndex
        numAgents = len(self.agents)

        for state in self.state.agentStates:
            print(state)

        self.display.initialize(self.state)

        while not self.gameOver:
            # self.display.update()
            # Execute the action
            agent = self.agents[agentIndex]

            observation = self.state.deepCopy()
            action: Action = agent.getAction(observation)
            assert isinstance(
                action, Action), "action must be an Action object"
            self.moveHistory.append((agentIndex, action))
            # try:
            #     self.state = self.state.getNextState(
            #         agentIndex, action)
            # except Exception as data:
            #     print("Something wrong happens")
            #     break
            self.state = self.state.getNextState(agentIndex, action)

            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)
            # Track progress
            if agentIndex == numAgents + 1:
                self.numMoves += 1
            # Next agent
            agentIndex = (agentIndex + 1) % numAgents
            self.display.update(self.state)
        pygame.quit()


def runGames(display: type, layout: Layout, player: Agent, ghosts: list[Agent], numGames: int = 1, numTraining: int = 0, catchExceptions: bool = False):

    rules = ClassicGameRules()
    games = []

    # 这里可以加一段，来使训练时没有图形界面
    for i in range(numGames):
        beQuiet = i < numTraining
        if beQuiet:
            # Suppress output and graphics
            from displayModes import NullGraphics
            gameDisplay = NullGraphics()
            rules.quiet = True
        else:
            gameDisplay = display(layout.map_size, layout.tile_size)
        game = rules.newGame(layout, player, ghosts,
                    gameDisplay, False, catchExceptions)
        gameDisplay = display(layout.map_size, layout.tile_size)
        rules.quiet = False
        game = rules.newGame(layout, player, ghosts, gameDisplay, False, catchExceptions)
        game.run()
        if not beQuiet:
            games.append(game)

        # if record:
        #     import time
        #     import pickle
        #     fname = ('recorded-game-%d' % (i + 1)) + \
        #         '-'.join([str(t) for t in time.localtime()[1:6]])
        #     f = file(fname, 'w')
        #     components = {'layout': layout, 'actions': game.moveHistory}
        #     pickle.dump(components, f)
        #     f.close()

    if (numGames-numTraining) > 0:
        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print(f'Win Rate:      {wins.count(True)}/{len(wins)} ({winRate:.2f})')
        print('Record:       ', ', '.join(['Loss', 'Win'][int(w)] for w in wins))

    return games
