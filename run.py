from ghostAgents import GreedyGhostAgent, GhostAgentSlightlyRandom, SmartGhostsAgent, GhostsAgentSample
from playerAgents import RandomAgent
from multiAgents import TimidAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, ApproximateQAgent
from searchAgents import MaxScoreAgent
from display import NullGraphics
from layout import Layout
from util import Vector2d,random
import pickle
import argparse
from layoutGenerator import SpecialLayoutGenerator, RandomLayoutGenerator
from game import Agent, Game, ClassicGameRules
import numpy as np


import pkgutil
if pkgutil.find_loader("rich"):
    from rich import traceback, print
    traceback.install()
if pkgutil.find_loader("torch"):
    from deepLearningAgents import DQNAgent
if pkgutil.find_loader("pygame"):
    from gui import PygameKeyboardAgent
    from gui import PygameGraphics
if pkgutil.find_loader("textual"):
    from tui import TextualKeyboardAgent
    from tui import TextualGraphics



def runGames(
    display: type,
    layout: Layout,
    player: Agent,
    ghosts: list[Agent],
    numGames: int = 1,
    scoreChange: list[int] = [1,125,750,-500],
    handleKeyboardInterrupt = True,
) -> tuple[list[Game], dict[str, float]]:
    rules = ClassicGameRules()
    games = []

    # 警告，判断一下ghost数量是不是等于ghost agent数量
    assert len(ghosts) == layout.getNumGhosts()

    gameDisplay = display(layout.map_size, layout.tile_size)

    try:
        for i in range(numGames):
            print(">>> Start game", i)
                    
            if layout.player_pos == Vector2d(-1,-1):
                if len(layout.ghosts_pos) == 0:
                    layoutGenerator = SpecialLayoutGenerator() if issubclass(type(player), DQNAgent) else RandomLayoutGenerator()
                    layout_ = layoutGenerator.generate(layout.map_size,layout.ghost_num)
                # else:
                #     layout.player_pos = Vector2d(*np.random.randint(1, layout.map_size.x+1, 2))

            layout_.arrangeAgents(layout_.player_pos, layout_.ghosts_pos)
            game = rules.newGame(layout_, player, ghosts, gameDisplay, scoreChange ,False)
            game.run()
            games.append(game)
    except KeyboardInterrupt as e:
        if handleKeyboardInterrupt:
            print(">>> Exit with KeyboardInterrupt")
            gameDisplay.finish()
        else:
            raise e
    finally:
        if i > 0:
            scores = [game.state.getScore() for game in games]
            wins = [game.state.isWin() for game in games]
            winRate = wins.count(True) / float(len(wins))
            averageScore = sum(scores) / float(len(scores))
            averageMoves = sum(game.numMoves for game in games) / float(len(games))
            print(f"Average Score: {averageScore}")
            print(f"Win Rate: {wins.count(True)}/{len(wins)} ({winRate:.2f})")
            print(f"Avg. Moves: {averageMoves:5.1f}")
            
            return games, {
                "winRate": winRate,
                "averageScore": averageScore,
                "averageMoves": averageMoves
            }




def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-size", action="store", type=int, nargs=2, default=[15, 15],
                        metavar=("WIDTH", "HEIGHT"),
                        help="The size of the map, default is 15 15.")
    parser.add_argument("--tile-size", action="store", type=int, nargs=2, default=[30, 30],
                        metavar=("WIDTH", "HEIGHT"),
                        help="The size of the tile, default is 30 30.")
    parser.add_argument("--ghost-num", action="store", type=int, default=4, 
                        help="The number of ghosts, default is 4.")
    parser.add_argument("--player-pos", action="store", type=int, nargs=2,
                        default=[-1, -1], metavar=("X", "Y"),
                        help="The position of the player, default is random.")
    parser.add_argument("--ghosts-pos", action="store", type=int, nargs="+", default=[],
                        help="The position of the ghosts, default is empty, which means random.")
    parser.add_argument("--player", action="store", type=str, default="PygameKeyboardAgent", 
                        choices=[
                            "RandomAgent", "TimidAgent", "AlphaBetaAgent", "AlphaBetaAgent",
                            "MCTSAgent", "MaxScoreAgent",
                            "QLearningAgent", "SarsaAgent", "SarsaLambdaAgent", "ApproximateQAgent",
                            "FullyConnectedDQNAgent", "OneHotDQNAgent", "ImitationAgent", "ActorCriticsAgent",
                            "FixnumPosDQNAgent",
                            "PygameKeyboardAgent",
                        ],
                        help="The agent to use, default is PygameKeyboardAgent.")
    parser.add_argument("--ghosts", action="store", type=str, default="GreedyGhostAgent", 
                        choices=[
                            "GreedyGhostAgent", "GhostAgentSlightlyRandom", "SmartGhostsAgent",
                        ],
                        help="The agent to use, default is SmartGhostsAgent.")
    parser.add_argument("--no-graphic", action="store_true", 
                        help="Whether to use graphic display, default is true.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Whether to print verbose information, default is false.")
    parser.add_argument("-n", "--num-of-games", action="store", type=int, default=1, 
                        help="The number of games to play, default is 1.")
    args = parser.parse_args()

    config = vars(args)
    config["map_size"] = Vector2d(*config["map_size"])
    config["tile_size"] = Vector2d(*config["tile_size"])
    config["player_pos"] = Vector2d(*config["player_pos"])
    config["ghosts_pos"] = list(map(Vector2d, zip(config["ghosts_pos"][::2], config["ghosts_pos"][1::2])))
    # config["ghosts_pos"] = list(map(lambda x: Vector2d(*x), zip(config["ghosts_pos"][::2], config["ghosts_pos"][1::2])))
    match config["player"]:
        case "RandomAgent":
            config["player"] = RandomAgent()
        case "TimidAgent":
            config["player"] = TimidAgent()
        case "AlphaBetaAgent":
            config["player"] = AlphaBetaAgent()
        case "ExpectimaxAgent":
            config["player"] = ExpectimaxAgent()
        case "MCTSAgent":
            config["player"] = MCTSAgent()
        case "MaxScoreAgent":
            config["player"] = MaxScoreAgent()
        case "QLearningAgent":
            config["player"] = pickle.load(open("QLearningAgent.pkl", "rb"))
        case "SarsaAgent":
            config["player"] = pickle.load(open("SarsaAgent.pkl", "rb"))
        case "SarsaLambdaAgent":
            config["player"] = pickle.load(open("SarsaLambdaAgent.pkl", "rb"))
        case "OneHotDQNAgent":
            config["player"] = pickle.load(open("OneHotDQNAgent.pkl", "rb"))
        case "FullyConnectedDQNAgent":
            config["player"] = pickle.load(open("FullyConnectedDQNAgent.pkl", "rb"))
        case "ImitationAgent":
            config["player"] = pickle.load(open("ImitationAgent.pkl", "rb"))
        case "ActorCriticsAgent":
            config["player"] = pickle.load(open("ActorCriticsAgent.pkl", "rb"))
        case "FixnumPosDQNAgent":
            config["player"] = pickle.load(open("FixnumPosDQNAgent.pkl", "rb"))
        case "PygameKeyboardAgent":
            config["player"] = PygameKeyboardAgent()
        case "ApproximateQAgent":
            config["player"] = pickle.load(open("ApproximateQAgent.pkl", "rb"))
        case _:
            raise ValueError(f"Unknown player agent {config['player']}")
    match config["ghosts"]:
        case "SmartGhostsAgent":
            config["ghosts"] = SmartGhostsAgent(config["ghost_num"])
        case "GhostsAgentSample":
            config["ghosts"] = GhostsAgentSample(config["ghost_num"])
        case "GreedyGhostAgent":
            config["ghosts"] = [GreedyGhostAgent(i) for i in range(1, config["ghost_num"]+1)]
        case "GhostAgentSlightlyRandom":
            config["ghosts"] = [GhostAgentSlightlyRandom(i) for i in range(1, config["ghost_num"]+1)]
        case _:
            raise ValueError(f"Unknown ghosts agent {config['ghosts']}")

    if config["no_graphic"]:
        config["display"] = NullGraphics
    else:
        config["display"] = PygameGraphics
    
    print(config)
    return config


if __name__ == "__main__":
    config = parse_args()
    layout = Layout(
        map_size=config["map_size"],
        tile_size=config["tile_size"],
        ghost_num=config["ghost_num"],
        player_pos=config["player_pos"],
        ghosts_pos=config["ghosts_pos"],
    )
    runGames(
        display=config["display"],
        layout=layout,
        player=config["player"],
        ghosts=config["ghosts"],
        numGames=config["num_of_games"],
    )