from game import runGames
from train import trainPlayer
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
from layoutGenerator import SpecialLayoutGenerator

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
                            "RandomAgent", "TimidAgent", "AlphaBetaAgent", "ExpectimaxAgent",
                            "MCTSAgent", "MaxScoreAgent",
                            "QLearningAgent", "SarsaAgent", "SarsaLambdaAgent", "ApproximateQAgent",
                            "FullyConnectedDQNAgent", "OneHotDQNAgent", "ImitationAgent", "ActorCriticsAgent",
                            "PygameKeyboardAgent", "TextualKeyboardAgent",
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
    parser.add_argument("-p", "--parallel", action="store", type=int, default=1, 
                        help="The maximum number of processes allowed.")
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
        case "PygameKeyboardAgent":
            config["player"] = PygameKeyboardAgent()
        case "TextualKeyboardAgent":
            config["player"] = TextualKeyboardAgent()
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
    
    if config['player_pos'] == Vector2d(-1,-1):
        if len(config['ghosts_pos']) == 0:
            layoutGenerator = SpecialLayoutGenerator()
            layout = layoutGenerator.generate(config['map_size'],config['ghost_num'])
            config["player_pos"] = layout.player_pos
            config["ghosts_pos"] = layout.ghosts_pos
        else:
            config['player_pos'] = Vector2d(random.randint(1, config['map_size'].x), random.randint(1, config['map_size'].y))
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