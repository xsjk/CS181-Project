from display import Display, NullGraphics
from environment import Environment, PlayerGameEnvironment
from util import ThreadTerminated, sign, Vector2d, Queue
from agentRules import Action, AgentState, Actions, Agent, Configuration, Direction
import traceback
from layout import Layout
import numpy as np
from threading import Thread
from rich.progress import track


class PlayerRules:
    """
    These functions govern how player interacts with his environment under
    the classic game rules.
    """

    @staticmethod
    def getLegalActions(state: "GameState"):
        """
        Returns a list of possible actions.d
        """
        return Actions.getPossibleActions(
            state.getPlayerState().configuration, state.layout
        )

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
            vector = Actions.actionToVector(action)

        playerState = state.agentStates[0]

        playerState.configuration = playerState.configuration.getNextState(vector)


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
        if Action.STOP in possibleActions:
            possibleActions.remove(Action.STOP)
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
        vector = Actions.actionToVector(action)
        ghostState.configuration = ghostState.configuration.getNextState(vector)

    @staticmethod
    def checkDeath(state: "GameState"):
        playerPosition = state.getPlayerPosition()
        # check if player is dead
        for index in range(1, len(state.agentStates)):
            GhostRules.checkOneDeath(state, index)
            # print("ghosts collides")
        state.updateScore(2)

    @staticmethod
    def checkOneDeath(state: "GameState", index: int):
        playerPosition = state.getPlayerPosition()
        ghostState = state.agentStates[index]
        ghostPosition = ghostState.getPosition()
        if GhostRules.canKill(playerPosition, ghostPosition):
            state.updateScore(3)
        if not state.agentStates[index].dead:
            # check if a ghost will boom with another ghost
            for i in range(1, len(state.agentStates)):
                if i != index:
                    if GhostRules.canKill(
                        state.agentStates[i].getPosition(),
                        state.agentStates[index].getPosition(),
                    ):
                        GhostRules.boom(
                            state, state.agentStates[i], state.agentStates[index]
                        )
                        state.updateScore(1)
                        return

    @staticmethod
    def canKill(playerPosition: Vector2d, ghostPosition: Vector2d) -> bool:
        return playerPosition == ghostPosition

    # @staticmethod
    # def collide(state: "GameState"):
    #     if not state._win and not state._lose:
    #         state.score -= 500
    #         state._lose = True

    @staticmethod
    # Notice that the boom will be called twice if a boom happen.
    def boom(
        state: "GameState", ghost_state1: "AgentState", ghost_state2: "AgentState"
    ):
        ghost_state1.dead = True
        ghost_state2.dead = True

    # @staticmethod
    # def checkWin(state):
    #     # if win
    #     num = sum(
    #         [state.agentStates[i].dead == False for i in range(1, state.getNumAgents())]
    #     )
    #     if not state._lose and num == 0 and not state._win:
    #         state.score += 750
    #         state._win = True

    @staticmethod
    def placeGhost(state: "GameState", ghostState: "AgentState"):
        ghostState.configuration = ghostState.start


class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def newGame(
        self,
        layout: Layout,
        playerAgent: Agent,
        ghostAgents: list[Agent],
        display,
        scoreChange: list[int],
        quiet: bool = False,
    ):
        # print(ghostAgents)
        agents = [playerAgent] + ghostAgents
        initState = GameState()
        initState.initialize(layout, agents, scoreChange)
        game = Game(agents, display, self)
        game.state = initState
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
            print(f"Player emerges victorious! Score: {state.score}")
        game.gameOver = True

    def lose(self, state: "GameState", game: "Game"):
        if not self.quiet:
            print(f"Player died! Score: {state.score}")
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
        self.scoreChange: list = []
        self.score = 0
        self.layout: Layout
        self.actionsTaken = []
        self.deadGhosts = 0

        if prevState != None:
            self.agentStates = self.copyAgentStates(prevState.agentStates)
            self.score = prevState.score
            self.layout = prevState.layout
            self.scoreChange = prevState.scoreChange
            self.agents = prevState.agents
            self.actionsTaken = prevState.actionsTaken
            self.deadGhosts = prevState.deadGhosts

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

    def initialize(self, layout: Layout, agents: list[Agent], scoreChange: list[int]):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        # self.capsules = []
        self.score = 0
        self.scoreChange = scoreChange

        self.agentStates = []
        numGhosts = 0

        self.layout = layout
        self.agents = agents

        pos = layout.agentPositions[0]
        self.agentStates.append(
            AgentState(Configuration(pos, Direction.NORTH.vector), True)
        )
        for i in range(1, layout.ghost_num + 1):
            pos = layout.agentPositions[i]
            self.agentStates.append(
                AgentState(Configuration(pos, Direction.NORTH.vector), False)
            )

    # static variable keeps track of which states have had getLegalActions called
    explored = set()

    @staticmethod
    def getAndResetExplored():
        tmp = GameState.explored.copy()
        GameState.explored = set()
        return tmp

    def getLegalActions(self, agentIndex: int = 0) -> list[Action]:
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
    
    def getLegalActionIndices(self, agentIndex: int = 0):
        """
        Returns the legal actions for the agent specified.
        """
        actions = self.getLegalActions(agentIndex)
        return [a.index for a in actions]

    def getLikelyActions(self, agentIndex: int = 1):
        """
        Returns the more possible actions for the ghost specified.
        """
        legal = GhostRules.getLegalActions(self, agentIndex)
        player_pos = self.getPlayerPosition()
        ghost_pos = self.getGhostPosition(agentIndex)
        dir_x = sign(player_pos.x - ghost_pos.x)
        # dir_y = sign(player_pos.y - ghost_pos.y)
        poss_actions = []
        for i in range(0, len(legal)):
            if dir_x == Actions.actionToVector(legal[i]).x:
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
        act = Actions.vectorToAction(Vector2d(dir_x, dir_y))
        return act

    def getLegalPlayerActions(self) -> list[Action]:
        return self.getLegalActions(0)

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
        if state.isLose():
            return state

        # First check if the player is dead

        # Then we update the remaind agents: ghosts
        actions = []
        for ghost in self.agents[1:]:
            actions.append(ghost.getAction(state))

        # print("The ghost action here is", action)
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
        dir = state.getPlayerState().getDirection()
        assert action == Action.from_vector(state.getPlayerState().getDirection())
        state.updateScore(0)

        playerPosition = state.getPlayerPosition()

        for index in range(1, len(state.agentStates)):
            ghostState = state.agentStates[index]
            ghostPosition = ghostState.getPosition()
            if GhostRules.canKill(playerPosition, ghostPosition):
                state.updateScore(3)

        self.actionsTaken.append(action)

        return state

    def getGhostsNextState(self, actions: list[Action]):
        """
        Returns the successsor state after the specified ghost actions( The player may not moved now! )
        """
        state = GameState(self)
        if len(actions) != self.getGhostNum():
            raise Exception("actions not right")
        for i in range(len(actions)):
            GhostRules.applyAction(state, actions[i], i + 1)
        GhostRules.checkDeath(state)
        return state
        # print("The ghost action here is", action)

    def getGhostNextState(self, action: Action, index: int):
        """
        Returns the successsor state after the specified ghost actions( The player may not moved now! )
        """
        state = GameState(self)
        GhostRules.applyAction(state, action, index)
        GhostRules.checkOneDeath(state, index)
        GhostRules.checkWin(state)
        return state

    def changeToNextState(self, action: Action):
        self.changeToPlayerNextState(action)
        if self.isLose():
            return
        # First check if the player is dead
        # Then we update the remaind agents: ghosts
        actions = [ghost.getAction(self) for ghost in self.agents[1:]]
        # print("The ghost action here is", action)
        self.changeToGhostsNextState(actions)

    def changeToGhostsNextState(self, actions: list[Action]):
        if len(actions) != self.getGhostNum():
            raise Exception("actions not right")
        for i in range(len(actions)):
            GhostRules.applyAction(self, actions[i], i + 1)
        GhostRules.checkDeath(self)
        # print("The ghost action here is", action)

    def changeToPlayerNextState(self, action: Action):
        """
        Generates the successor state after the specified player move
        """
        PlayerRules.applyAction(self, action)
        self.updateScore(0)
        self.actionsTaken.append(action)

        playerPosition = self.getPlayerPosition()

        for index in range(1, len(self.agentStates)):
            ghostState = self.agentStates[index]
            ghostPosition = ghostState.getPosition()
            if GhostRules.canKill(playerPosition, ghostPosition):
                self.updateScore(3)

    def getSuccessors(self)-> list:
        """
        Generates the successor states after any legal actions
        """
        legal_actions = self.getLegalActions()
        next_states = []
        for action in legal_actions:
            next_states.append(self.getNextState(action))

        return next_states

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
    
    def updateScore(self, index:int) -> int:
        '''
        index 0-3: normal,kill_ghost,win,lose
        
        '''
        if(index == 0):
            self.score += self.scoreChange[0]
        elif(index == 1):
            self.score += self.scoreChange[1]
        elif(index == 2):
            num = sum(
                [self.agentStates[i].dead == False for i in range(1, self.getNumAgents())]
            )
            if not self._lose and num == 0 and not self._win:
                self.score += self.scoreChange[2]
                self._win = True
        elif(index == 3):
            if not self._win and not self._lose:
                self.score += self.scoreChange[3]
                self._lose = True
        else:
            NotImplementedError("The index is out of range")

    def getActionsTaken(self) -> list[Action]:
        return self.actionsTaken

    def getActionsNum(self) -> int:
        return len(self.actionsTaken)

    def getDeadNum(self) -> int:
        '''
        Returns the number of the dead ghosts.

        '''
        return sum(
         [self.agentStates[i].dead == True for i in range(1, self.getNumAgents())]
        )

    def getDetecReward(self) -> float:
        '''
        Return a reward of the current state using the position detection.
        
        '''
        if self.isLose(): return 0.0
        rewards:float = 0.0
        # rewards += self.getDeadNum()
        player_pos = self.getPlayerPosition()
        
        lay_dis = abs(player_pos.x - self.layout.width//2) + abs(player_pos.y - self.layout.height//2)
        rewards -= lay_dis / (self.layout.width + self.layout.height)

        dead_num = self.getDeadNum()
        rewards += (dead_num - self.deadGhosts)*20

        for i in range(1,self.getNumAgents()):
            ghost_pos = self.getGhostPosition(i)
            distance = Vector2d.manhattanDistance(player_pos,ghost_pos)
            if(self.agentStates[i].dead == False):
                # 最开始先计算鬼离人的距离
                if(distance <= 2): 
                    rewards -= 100
                    break
                rewards += distance
                for j in range(i+1,self.getNumAgents()):
                    ghost_pos2 = self.getGhostPosition(j)
                    # check if the ghosts are close to each other
                    if(abs(ghost_pos.x - ghost_pos2.x) == 1 and abs(ghost_pos.y - ghost_pos2.y) == 1):
                        rewards += 5
                    # 鬼同列
                    elif(ghost_pos.y == ghost_pos2.y):
                        # 当和玩家同列
                        if(ghost_pos.y == player_pos.y):
                            # if another ghost is alive
                            if(self.agentStates[j].dead == False): rewards -= 10
                            # if another ghost is dead 
                            else: rewards += 10
                        # 当不和玩家同列
                        else: rewards += 10
                    # 鬼同行
                    elif(ghost_pos.x == ghost_pos2.x):
                        # 当和玩家同行
                        if(ghost_pos.x == player_pos.x):
                            # if another ghost is alive
                            if(self.agentStates[j].dead == False): rewards -= 10
                            # if another ghost is dead 
                            else: rewards += 10
                        # 当不和玩家同行
                        else: rewards += 10
        return rewards
        
    def getBfsReward(self,depth:int = 2) -> float:
        '''
        Return a reward of the current state using the position detection.
        
        '''    
        from util import Queue
        score = self.getScore()
        scores = []
        actions_num = len(self.actionsTaken)
        start_state = GameState(self)
        states = Queue()

        states.push(start_state)

        while states.isEmpty() != True:
            state = states.pop()
            if(state.isWin() or state.isLose() or
            len(state.actionsTaken) - actions_num > depth):
                scores.append(state.getScore())
                continue
            for next_state in state.getSuccessors():
                states.push(next_state)
        
        return max(scores) - score



    def isLose(self) -> bool:
        return self._lose

    def isWin(self) -> bool:
        return self._win

    def toOneHotMatrix(self):
        """
        Returns a one-hot matrix representation of the game state
        shape: (3, height, width)

        """
        mat = np.zeros((3, self.layout.height, self.layout.width), dtype=int)
        # 0: player
        # 1: ghost
        # 2: dead ghost
        player_pos = self.getPlayerPosition()
        mat[0][player_pos.y - 1][player_pos.x - 1] = 1
        for ghost in self.getGhostStates():
            pos = ghost.getPosition()
            if ghost.dead:
                mat[2][pos.y - 1][pos.x - 1] = 1
            else:
                mat[1][pos.y - 1][pos.x - 1] = 1
        return mat

    def toMatrix(self):
        """
        Returns a matrix representation of the game state
        shape: (height, width)

        """
        # 0: empty
        # 1: player
        # 2: ghost
        # 3: dead ghost
        mat = np.zeros((self.layout.height, self.layout.width), dtype=int)
        player_pos = self.getPlayerPosition()
        mat[player_pos.y - 1][player_pos.x - 1] = 1
        for ghost in self.getGhostStates():
            pos = ghost.getPosition()
            if ghost.dead:
                mat[pos.y - 1][pos.x - 1] = 3
            else:
                mat[pos.y - 1][pos.x - 1] = 2
        return mat


# The main controll flow


class Game:
    agents: list[Agent]
    display: Display
    rules: ClassicGameRules
    state: GameState
    gameOver: bool = False
    gameThread: Thread

    def __init__(
        self, agents: list[Agent], display, gameRule: ClassicGameRules
    ):
        # pygame.init()
        self.display = display

        self.agents = agents
        self.rules = gameRule

        self.moveHistory: list[tuple[int, Action]] = []
        self.score = 0
        self.gameThread = Thread(target=self.gameLoop)

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


        self.display.initialize(self.state)
        self.gameThread.start()
        if not isinstance(self.display, NullGraphics):
            while not self.gameOver and self.display.running:
                self.display.update(self.state)
        
        self.display.finish()
        self.gameThread.join()

    def gameLoop(self) -> None:
        try:
            while not self.gameOver:
                player: Agent = self.agents[0]
                action: Action = player.getAction(self.state)
                self.state.changeToNextState(action)
                self.numMoves += 1
                self.rules.process(self.state, self)
                print("The reward is",self.state.getBfsReward(3),"\n")
        except ThreadTerminated:
            self.updateScore(3)
            pass
        print(">>> Game loop finished")

def runGames(
    display: type,
    layout: Layout,
    player: Agent,
    ghosts: list[Agent],
    numGames: int = 1,
    scoreChange: list[int] = [1,125,750,-500],
):
    rules = ClassicGameRules()
    games = []

    # 警告，判断一下ghost数量是不是等于ghost agent数量
    assert len(ghosts) == layout.getNumGhosts()

    gameDisplay = display(layout.map_size, layout.tile_size)

    try:
        for i in range(numGames):
            print(">>> Start game", i)
            layout.arrangeAgents(layout.player_pos, layout.ghosts_pos)
            game = rules.newGame(layout, player, ghosts, gameDisplay, scoreChange ,False)
            game.run()
            games.append(game)
    except KeyboardInterrupt:
        print(">>> Exit with KeyboardInterrupt")
        gameDisplay.finish()
    finally:
        if i > 0:
            scores = [game.state.getScore() for game in games]
            wins = [game.state.isWin() for game in games]
            winRate = wins.count(True) / float(len(wins))
            print(f"Average Score: {sum(scores) / float(len(scores))}")
            # print("Scores:       ", ", ".join([str(score) for score in scores]))
            print(f"Win Rate: {wins.count(True)}/{len(wins)} ({winRate:.2f})")
            # print("Record:       ", ", ".join(["Loss", "Win"][int(w)] for w in wins))
            print(f"Avg. Moves: {sum(game.numMoves for game in games) / float(len(games)):5.1f}")
            
    return games



def trainPlayer(
    displayType: type,
    layout: Layout,
    player: Agent,
    ghosts: list[Agent],
    envType: type = PlayerGameEnvironment,
    numTrain: int = 100,
    scoreChange: list[int] = [1,125,750,-500],
):
    rules = ClassicGameRules()

    display = displayType(layout.map_size, layout.tile_size)
    player.epsilon = 1.0

    for _ in track(range(numTrain), description="Training..."):
        layout.arrangeAgents(layout.player_pos, layout.ghosts_pos)
        # print(layout.agentPositions)
        game: Game = rules.newGame(
            layout, player, ghosts, display, scoreChange, 
        )
        env: Environment = envType(player, startState=game.state)
        player.train(env)

        scores = env.state.getScore()
        wins = env.state.isWin()
        print(f"Score: {scores}, Win: {wins}, Epsilon: {player.epsilon}")
    player.epsilon = 0.0
    return player
