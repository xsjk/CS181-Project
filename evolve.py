from evolution import GA
from reinforcementAgents import ApproximateQAgent, MyFeatures
import pickle
from display import NullGraphics
from ghostAgents import GreedyGhostAgent
from game import runGames
from layout import Layout
from util import Vector2d

if __name__ == "__main__":

    map_size = Vector2d(15, 15)

    featureExtractor = MyFeatures()
    feature_names = featureExtractor.feature_names
    playerAgent = ApproximateQAgent(featureExtractor)
    ghostsAgent = [GreedyGhostAgent(i) for i in range(1, 6)]
    numGamesPerEvaluation = 100
    layout = Layout(
        map_size = map_size,
        tile_size = (30,30),
        ghost_num = 5,
        player_pos = None,
        ghosts_pos = [],
    )


    def evaluator(params) -> float:
        playerAgent.setWeights(params)
        _, result = runGames(
            display=NullGraphics,
            layout=layout,
            player=playerAgent,
            ghosts=ghostsAgent,
            numGames=numGamesPerEvaluation,
            handleKeyboardInterrupt=False
        )
        score = result["averageScore"]
        # score = result["winRate"]
        # score = result["averageMoves"]
        return score
    
    ea = GA(
        DNA_idx=feature_names, 
        fitnessEvaluator=evaluator, 
        pop_size=50, 
        n_kid=75, 
        times=20
    )
    ea.initialize()
    # ea = pickle.load(open("GA.pkl", "rb"))

    try:
        ea.mainloop()
    except KeyboardInterrupt:
        print("Evolution stopped by user")
    finally:
        # TODO save the population to file
        pickle.dump(ea, open("GA.pkl", "wb"))

