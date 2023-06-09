from display import Display, NullGraphics
from util import ThreadTerminated, sign, Vector2d
from agentRules import Action, AgentState, Actions, Agent, Configuration, Direction
import traceback
from layout import Layout
import numpy as np
from threading import Thread
from rich.progress import track
import math


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
        elif action == Action.UNDO:
            raise NotImplementedError
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
            print(f"Player emerges victorious! Score: {state.score+500}")
        game.gameOver = True

    def lose(self, state: "GameState", game: "Game"):
        if not self.quiet:
            print(f"Player died! Score: {state.score+500}")
        game.gameOver = True

    def agentCrash(self, game: "Game", agentIndex: int):
        if agentIndex == 0:
            print("Player crashed")
        else:
            print("A ghost crashed")


class GameState:

    _lose: bool = False
    _win: bool = False

    agentStates: list[AgentState]
    scoreChange: list[float] = []
    score: float = 0
    layout: Layout
    actionsTaken: list[Action] = []
    deadGhosts: int = 0

    def __init__(self, prevState=None):
        """
        Generates a new data packet by copying information from its predecessor.
        """

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
        for state in self.agentStates:
            try:
                int(hash(state))
            except TypeError as e:
                print(e)
                # hash(state)
        return hash(tuple(self.agentStates))

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

    def getAliveGhostPositions(self) -> list[Vector2d]:
        return [s.getPosition() for s in self.getGhostStates() if not s.dead]

    def getDeadGhostPositions(self) -> list[Vector2d]:
        return [s.getPosition() for s in self.getGhostStates() if s.dead]

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

    def getDetectReward(self) -> float:
        '''
        Return a reward of the current state using the position detection.
        
        '''
        import math
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
                # if(distance <= 2): 
                #     rewards -= 100
                #     break
                rewards -= 20/math.exp(distance) 
                for j in range(i+1,self.getNumAgents()):
                    ghost_pos2 = self.getGhostPosition(j)
                    # check if the ghosts are close to each other
                    if(abs(ghost_pos.x - ghost_pos2.x) == 1 and abs(ghost_pos.y - ghost_pos2.y) == 1):
                        rewards += 5
                    # 鬼同列
                    if(ghost_pos.y == ghost_pos2.y):
                        # 当和玩家同列
                        if(ghost_pos.y == player_pos.y):
                            # if another ghost is alive
                            if(self.agentStates[j].dead == False): rewards -= 5
                            # if another ghost is dead 
                            else: rewards += 5
                        # 当不和玩家同列
                        else: rewards += 5
                    # 鬼同行
                    if(ghost_pos.x == ghost_pos2.x):
                        # 当和玩家同行
                        if(ghost_pos.x == player_pos.x):
                            # if another ghost is alive
                            if(self.agentStates[j].dead == False): rewards -= 5
                            # if another ghost is dead 
                            else: rewards += 5
                        # 当不和玩家同行
                        else: rewards += 5
        return rewards
        
    def getBFSReward(self,depth:int = 2) -> float:
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

    def getCluster(self,ghost_states:list[AgentState])->tuple[list[AgentState],list[AgentState]]:
        large_list:list[AgentState]
        small_list:list[AgentState]
        dirs = [Actions.actionToVector(self.getGreedyAction(i)) for i in range(1,self.getGhostNum()+1)]
        ghostnum = len(ghost_states)
        # calculate the pattern
        pattern_x,pattern_y = (0,0)
        for i in dirs:
            pattern_x += i.x
            pattern_y += i.y

        # 4 + 0
        if(abs(pattern_x) == ghostnum or abs(pattern_y) == ghostnum):
            print("按行或列分4个\n")
            large_list = ghost_states
            small_list = []
        # 1 + 3
        elif(abs(pattern_x) - ghostnum >= -2):
            print("按列分3个\n")
            large_list = list(filter(lambda key:key.getDirection().x*pattern_x >= 0,ghost_states))
            small_list = list(filter(lambda key:key.getDirection().x*pattern_x <= 0,ghost_states))
            pass
        elif(abs(pattern_y) - ghostnum >= -2):
            print("按行分3个\n")
            large_list = list(filter(lambda key:key.getDirection().y*pattern_y >= 0,ghost_states))
            small_list = list(filter(lambda key:key.getDirection().y*pattern_y <= 0,ghost_states))
        elif(abs(pattern_x) - ghostnum >=-4):
            print("按列分2个\n")
            large_list = list(filter(lambda key:key.getDirection().x*pattern_x >= 0,ghost_states))
            small_list = list(filter(lambda key:key.getDirection().x*pattern_x <= 0,ghost_states))
        # 2+2
        elif(abs(pattern_y) - ghostnum >= -4):
            print("按行分2个\n")
            large_list = list(filter(lambda key:key.getDirection().y >= 0,ghost_states))
            small_list = list(filter(lambda key:key.getDirection().y <= 0,ghost_states))
        
        return large_list,small_list 

    def getPatternReward(self) -> float:
        '''
        Calculate the reward based on the pattern of the world.
        There are totally 4 patterns:
        1. all ghost in one corner 2. 1+3 ghost position 3. 2+2 ghost position 4. ghost are besides the player
        '''

        def getCircle(ghost_states:list[AgentState])->float:
            circle = 0.0
            for i in range(len(ghost_states)):
                for j in range(i+1,len(ghost_states)): 
                    circle += Vector2d.manhattanDistance(ghost_states[i].getPosition(),
                                                         ghost_states[j].getPosition())
            return circle

        if(self.isLose() or self.isWin()): return 0

        # define the variables we need
        rewards = 0.0
        ghost_states:list[AgentState] = self.getGhostStates()
        large_states,small_states = self.getCluster(ghost_states)
        player_pos = self.getPlayerPosition()
        all_live_dis:list[float] = []
        all_dead_dis = list[float]()
        all_dis = list[float]()
        clo_live_dis:float
        clo_dead_dis:float
        clo_dis:float 
        circle_dis:float = getCircle(large_states)
        ghostnum = self.getGhostNum()
        
        for state in large_states:
            dis = Vector2d.manhattanDistance(state.getPosition(),player_pos)
            all_dis.append(dis)
            if(state.dead): all_dead_dis.append(dis)
            else: all_live_dis.append(dis)
        clo_live_dis = min(all_live_dis)
        clo_dead_dis = min(all_dead_dis,default=None)

        if(clo_live_dis <= 1):
            rewards -= 10
        # all ghost in one corner
        if(len(large_states) == ghostnum):
            rewards += 10
            rewards -= 20/math.exp(clo_live_dis) 
            if(clo_dead_dis != None): rewards += 10/math.exp(clo_dead_dis)
            rewards += 40/circle_dis            
        # 3 + 1
        elif(len(large_states) == ghostnum-1):
            rewards += 5
            single_state = small_states[0]
            single_dis =  Vector2d.manhattanDistance(single_state.getPosition(),player_pos)
            if(not single_state.dead): rewards -= 15/math.exp(single_dis) 
            
            rewards -= 10/math.exp(clo_live_dis) 
            if(clo_dead_dis != None): rewards += 10/math.exp(clo_dead_dis)
            rewards += 40/circle_dis   
        # 2+2
        elif(len(large_states) == ghostnum-2):
            circle_dis2 = getCircle(small_states)
            if(circle_dis2 == 0): circle_dis2 += 20
            rewards += 20/(circle_dis2)

            rewards -= 10/math.exp(clo_live_dis) 
            if(clo_dead_dis != None): rewards += 10/math.exp(clo_dead_dis)
            rewards += 40/circle_dis 
        # besides
        return rewards


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
        self.history = []
        try:
            while not self.gameOver:
                player: Agent = self.agents[0]
                action: Action = player.getAction(self.state)
                if action == Action.UNDO:
                    if self.history == []:
                        print("history is empty!")
                    else:
                        self.state = self.history.pop()
                else:
                    self.history.append(self.state.deepCopy())
                    self.numMoves += 1
                    self.state.changeToNextState(action)
                    self.rules.process(self.state, self)
                # print("The reward of that state is ",self.state.getPatternReward(),"\n")
        except ThreadTerminated:
            self.state.updateScore(3)
            pass
