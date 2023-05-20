from game import *

class mousePlayer(Agent):
    def __init__(self, index:int = 0):
        super(mousePlayer, self).__init__(index)
        self.color = COLORS["player"]
    
    # 通过键盘获取移动
    def getAction(self, state:GameState):
        action = None
        while action == None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    else:
                        for act in Directions.VALID_ACTIONS:
                            if event.key in Directions.ACTION_KEYS[act]:
                                action = act
                                break
        if(action):
            assert action in Directions.VALID_ACTIONS, f"move action \"{action}\" is invalid"
            for a in action:
                match a:
                    case "N":
                        return Directions.NORTH
                    case "S":
                        return Directions.SOUTH
                    case "W":
                        return Directions.WEST
                    case "E":
                        return Directions.EAST
        return Directions.WEST
        

