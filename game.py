from time import sleep
from environment import Environment, PlayerGameEnvironment
from util import *
from agentRules import *
import traceback
from layout import Layout
import numpy as np
from displayModes import GraphicMode
from rich.progress import track


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
                assert "1234"
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
    def checkDeath(state: "GameState"):
        playerPosition = state.getPlayerPosition()
        score_change = 0
        # check if player is dead
        for index in range(1, len(state.agentStates)):
            ghostState = state.agentStates[index]
            ghostPosition = ghostState.getPosition()
            if GhostRules.canKill(playerPosition, ghostPosition):
                GhostRules.collide(state)
            if not state.agentStates[index].dead: 
            # check if a ghost will boom with another ghost 
                for i in range(1, len(state.agentStates)):
                    if(i != index):
                        if GhostRules.canKill(state.agentStates[i].getPosition(), state.agentStates[index].getPosition()):
                            GhostRules.boom(state,state.agentStates[i],state.agentStates[index])
                            score_change += 125
                            break
                    # print("ghosts collides")
        
        # if win
        num = sum([state.agentStates[i].dead == False for i in range(1,state.getNumAgents())])
        #state.score += 125
        if not state._lose:
            state.score += score_change
        
        if not state._lose and num == 0 and not state._win:
            state.score += 750
            state._win = True

    @staticmethod
    def canKill(playerPosition: Vector2d, ghostPosition: Vector2d) -> bool:
        return playerPosition == ghostPosition

    @staticmethod
    def collide(state):
        if not state._win and not state._lose:
            state.score -= 500
            state._lose = True
    
    @staticmethod
    # Notice that the boom will be called twice if a boom happen.
    def boom(state,ghost_state1,ghost_state2):
        ghost_state1.color = COLOR["explosion"]
        ghost_state1.dead = True
        ghost_state2.color = COLOR["explosion"]
        ghost_state2.dead = True


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
        initState.initialize(layout, agents)
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
            print("Player emerges victorious! Score: %d" % state.score)
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
        self.actionsTaken = []

        if prevState != None:
            self.agentStates = self.copyAgentStates(prevState.agentStates)
            self.score = prevState.score
            self.layout = prevState.layout
            self.scoreChange = prevState.scoreChange
            self.agents = prevState.agents
            self.actionsTaken = prevState.actionsTaken

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

    def initialize(self, layout: Layout, agents:list[Agent]):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        # self.capsules = []
        self.score = 0
        self.scoreChange = 1

        self.agentStates = []
        numGhosts = 0

        self.layout = layout
        self.agents = agents

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
    
    def getLikelyActions(self, agentIndex: int = 1):
        """
        Returns the more possible actions for the ghost specified.
        """
        legal = GhostRules.getLegalActions(self, agentIndex)
        player_pos = self.getPlayerPosition()
        ghost_pos = self.getGhostPosition(agentIndex)
        dir_x = sign(player_pos.x - ghost_pos.x)
        #dir_y = sign(player_pos.y - ghost_pos.y)
        poss_actions = []
        for i in range(0,len(legal)):
            if(dir_x == Actions.actionToVector(legal[i]).x):
                poss_actions.append(legal[i])
        return poss_actions
    
    def getGreedyAction(self, agentIndex: int = 1):
        """
        Returns the more possible actions for the ghost specified.
        """
        # legal = GhostRules.getLegalActions(self, agentIndex)
        player_pos = self.getPlayerPosition()
        ghost_pos = self.getGhostPosition(agentIndex)
        dir_x = sign(player_pos.x - ghost_pos.x)
        dir_y = sign(player_pos.y - ghost_pos.y)
        act = Actions.vectorToAction(Vector2d(dir_x,dir_y))
        return act

    def getLegalPlayerActions(self) -> list[Action]:
        return self.getLegalActions(0)

    @type_check
    def getNextState(self, action: Action) -> "GameState":
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.isWin() or self.isLose():
            raise Exception("Can't generate a successor of a terminal state.")

        # Let agent's logic deal with its action's effects on the board
        # First we update the player state
        state = self.getPlayerNextState(action)
        if(state.isLose()): return state


        # First check if the player is dead

        # Then we update the remaind agents: ghosts
        actions = []
        for ghost in self.agents[1:]:
            actions.append(ghost.getAction(state))
            
        #print("The ghost action here is", action)
        state = state.getGhostsNextState(actions)   

        # GameState.explored.add(self)
        # GameState.explored.add(state)

        return state 

    def getPlayerNextState(self, action: Action) -> "GameState":
        """
        Generates the successor state after the specified player move
        """
        state = GameState(self)
        PlayerRules.applyAction(state, action)
        state.score += state.scoreChange
        
        playerPosition = state.getPlayerPosition()

        for index in range(1, len(state.agentStates)):
            ghostState = state.agentStates[index]
            ghostPosition = ghostState.getPosition()
            if GhostRules.canKill(playerPosition, ghostPosition):
                GhostRules.collide(state)
        
        self.actionsTaken.append(action)

        return state
    
    def getGhostsNextState(self, actions:list[Action]):
        """
        Returns the successsor state after the specified ghost actions( The player may not moved now! ) 
        """
        state = GameState(self)
        if(len(actions) != self.getGhostNum()): raise Exception("actions not right")
        for i in range(len(actions)):
            GhostRules.applyAction(state, actions[i], i+1)
        GhostRules.checkDeath(state)
        return state
        #print("The ghost action here is", action)

    def getGhostNextState(self, action:Action, index:int):
        """
        Returns the successsor state after the specified ghost actions( The player may not moved now! ) 
        """
        state = GameState(self)
        GhostRules.applyAction(state, action, index)
        GhostRules.checkDeath(state)
        return state

        
    def changeToNextState(self, action: Action):
        self.changeToPlayerNextState(action)
        if(self.isLose()): return 
        # First check if the player is dead
        # Then we update the remaind agents: ghosts
        actions = []
        for agentIndex in range(1,self.getGhostNum()+1):
            actions.append(self.agents[agentIndex].getAction(self))
            #print("The ghost action here is", action)
        self.changeToGhostsNextState(actions)
    
    def changeToGhostsNextState(self, actions:list[Action]):
        if(len(actions) != self.getGhostNum()): raise Exception("actions not right")
        for i in range(len(actions)):
            GhostRules.applyAction(self, actions[i], i+1)
        GhostRules.checkDeath(self)
        #print("The ghost action here is", action)
   
    def changeToPlayerNextState(self, action: Action):
        """
        Generates the successor state after the specified player move
        """
        PlayerRules.applyAction(self, action)
        self.score += self.scoreChange
        self.actionsTaken.append(action)

        playerPosition = self.getPlayerPosition()
        

        for index in range(1, len(self.agentStates)):
            ghostState = self.agentStates[index]
            ghostPosition = ghostState.getPosition()
            if GhostRules.canKill(playerPosition, ghostPosition):
                GhostRules.collide(self)

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

    def getGhostNum(self) -> int:
        return len(self.agentStates) - 1

    def getMapSize(self) -> Vector2d:
        return self.layout.map_size

    def getScore(self) -> float:
        return float(self.score)
    
    def getActionsTaken(self) -> list[Action]:
        return self.actionsTaken

    def getActionsNum(self) -> int:
        return len(self.actionsTaken)

    def isLose(self) -> bool:
        return self._lose

    def isWin(self) -> bool:
        return self._win
    
    def toMatrix(self):
        mat = np.zeros((self.layout.height, self.layout.width), dtype=int)
        # 0: empty
        # 1: player
        # 2: ghost
        # 3: dead ghost
        player_pos = self.getPlayerPosition()
        mat[player_pos.y][player_pos.x] = 1
        for ghost in self.getGhostStates():
            pos = ghost.getPosition()
            if ghost.dead:
                mat[pos.y][pos.x] = 3
            else:
                mat[pos.y][pos.x] = 2
        return mat


# def gridToPixel(pos: tuple) -> Vector2d:
#     return (pos[0] * TILE_SIZE.width - TILE_SIZE.width // 2, pos[1] * TILE_SIZE.height - TILE_SIZE.height // 2)


def isOdd(x: int) -> bool:
    return bool(x % 2)

# The main controll flow


class Game:
    def __init__(self, agents: list[Agent], display: GraphicMode, gameRule: ClassicGameRules, catchExceptions):
        # pygame.init()
        self.display = display

        self.agents = agents
        self.rules = gameRule

        self.startingIndex = 0
        self.gameOver = False
        self.catchExceptions = catchExceptions
        self.moveHistory: list[tuple[int, Action]] = []
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
                print(f"Agent {i} failed to load", file=sys.stderr)
                self._agentCrash(i, quiet=True)
                return

        agentIndex = self.startingIndex
        numAgents = len(self.agents)

        # for state in self.state.agentStates:
        #     print(state)

        self.display.initialize(self.state)

        while not self.gameOver:
            sleep(1)
            # Execute the action
            agent = self.agents[agentIndex]

            #observation = self.state.deepCopy()
            action: Action = agent.getAction(self.state)
            assert isinstance(
                action, Action), "action must be an Action object"
            
            # self.moveHistory.append((agentIndex, action))
            self.state.changeToNextState(action)

            # Allow for game specific conditions (winning, losing, etc.)
            # Track progress
            self.rules.process(self.state, self)
            
            # Update the gui
            self.display.update(self.state)
            
        self.display.finish()


def runGames(display: type, layout: Layout, player: Agent, ghosts: list[Agent], numGames: int = 1, catchExceptions: bool = False):

    rules = ClassicGameRules()
    games = []

    # 警告，判断一下ghost数量是不是等于ghost agent数量
    assert len(ghosts) == layout.getNumGhosts()

    for i in range(numGames):
        gameDisplay = display(layout.map_size, layout.tile_size)
        rules.quiet = False
        game = rules.newGame(layout, player, ghosts, gameDisplay, False, catchExceptions)
        game.run()
        games.append(game)

        scores = [game.state.getScore() for game in games]
        wins = [game.state.isWin() for game in games]
        winRate = wins.count(True) / float(len(wins))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print(f'Win Rate:      {wins.count(True)}/{len(wins)} ({winRate:.2f})')
        print('Record:       ', ', '.join(['Loss', 'Win'][int(w)] for w in wins))
    return games


def trainPlayer(display: type, layout: Layout, player: Agent, ghosts: list[Agent], numTrain: int = 100, catchExceptions: bool = False):

    rules = ClassicGameRules()

    gameDisplay = display(layout.map_size, layout.tile_size)

    game: Game = rules.newGame(layout, player, ghosts, gameDisplay, False, catchExceptions)

    env: Environment = PlayerGameEnvironment(player, startState=game.state)

    for _ in track(range(numTrain), description='Training...'):
        player.train(env)
        
    # scores = [game.state.getScore() for game in games]
    # wins = [game.state.isWin() for game in games]
    # winRate = wins.count(True) / float(len(wins))
    # print('Average Score:', sum(scores) / float(len(scores)))
    # print('Scores:       ', ', '.join([str(score) for score in scores]))
    # print(f'Win Rate:      {wins.count(True)}/{len(wins)} ({winRate:.2f})')
    # print('Record:       ', ', '.join(['Loss', 'Win'][int(w)] for w in wins))


