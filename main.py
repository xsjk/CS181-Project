from game import runGames
from enemyAgents import normEnemy
from playerAgents import mousePlayer

if __name__ == "__main__":
    runGames(mousePlayer(),[normEnemy(i+1) for i in range(0,4)])
