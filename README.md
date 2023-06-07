*Python* >= 3.10 is required.
Notice: 
    1.It may takes some time to initialize, please be patient.
    2.You may need "pip install -r requirements.txt" to install all packages needed.

Usage:
    python run.py [options]


Example:
    python run.py 
    python run.py --player MaxScoreAgent --player-pos 7 7 --ghost-num 5
    python run.py --player RandomAgent -n 100 --no-graphics


The keyboard for keyboardAgent:
    -move                       W,A,S,D,Q,E,Z,X
    -stop                       P
    -transpose(only for keyboard) 
                                SPACE
    -undo                       U


General Options:
    -h, --help                  Show help.
    --map-size WIDTH HEIGHT     The size of the map, default is 15 15.
                                The origin starts from (1,1) in the left-top.

    --tile-size WIDTH HEIGHT    The size of the tile, default is 30 30.

    --ghost-num GHOST_NUM       The number of ghosts, it should not be too large to fufill the map.

    --player-pos X Y            The position of the player, default is random.

    --ghosts-pos GHOST_POS [GHOSTS_POS ...]
                                The position of the ghosts, the number it should be equal to your ghostnum.
                                Default is empty, which means random.
    
    --player                    The player agent to use, default is PygameKeyboadAgent
    Choices for that: {RandomAgent,TimidAgent,AlphaBetaAgent,ExpectimaxAgent,MCTSAgent,MaxScoreAgent,QLearningAgent,SarsaAgent,SarsaLambdaAgent,ApproximateQAgent,FullyConnectedDQNAgent,OneHotDQNAgent,ImitationAgent,ActorCriticsAgent,PygameKeyboardAgent,TextualKeyboardAgent}

    --ghosts                    The ghost agent to use, default is SmartGhostsAgent.
    Choices for that: {GreedyGhostAgent,GhostAgentSlightlyRandom,SmartGhostsAgent}

    --no-graphic                Whether to use graphic display, default is true. 
                                It could speed the process.
    
    -v, --verbos                Whether to print verbose information, default is false.

    -n NUM_OF_GAMES --num-of-games NUM_OF_GAMES
                                The number of games to play, default is 1.
    
    -p PARALLEL, --parallel PARALLEL
                                The maximum number of precesses allowed.


Explanation for the Player Agents:
    PygameKeyboardAgent:        Using the keyboard to control the agent, the default agent.

    RandomAgent:                The player moves to legal position randomly.

    TimidAgent:                 The greedy player always tries to escape from the ghosts.

    AlphaBetaAgent:             The player finds the best action based on the min-max tree.

    ExpectimaxAgent:            The player finds the best action based on the expect of min-max tree.

    MCTSAgent:                  Using the MCTS method to arrange the action for the player.

    MaxScoreAgent:              Using the BFS method to arrange the action for the player.

    Other Agents:               Using different ways of reinforcement learning to train the player.


Explanation for the Enemy Agents:
    GreedyGhostAgent：          The ghosts only consider chasing the player. The default agent.

    hostAgentSlightlyRandom:    The ghosts sometimes act randomly.

    SmartGhostsAgent:           The ghosts will try to aviod crashing into other ghosts.
























