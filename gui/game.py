from utils import *
import random


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.surface = pygame.display.set_mode(WINDOW_SIZE.sizeTuple)
        self.running = True
        self.player = Player(
            self.surface,
            MAP_SIZE.width // 2,
            MAP_SIZE.height // 2
        )
        self.enemies: list[Enemy] = []
        for _ in range(ENEMY_NUMBER):
            self.enemies.append(Enemy(
                self.surface,
                random.randint(1, MAP_SIZE.width),
                random.randint(1, MAP_SIZE.height)
            ))
        self.score = 0

    def updateScore(self) -> None:
        self.score += 1
        print("Score:", self.score)

    def drawBg(self) -> None:
        for x in range(MAP_SIZE.width):
            for y in range(MAP_SIZE.height):
                pygame.draw.rect(
                    self.surface,
                    COLORS["tileBg0"] if isOdd(x + y) else COLORS["tileBg1"],
                    (x * TILE_SIZE.width, y * TILE_SIZE.height, TILE_SIZE.width, TILE_SIZE.height)
                )

    def movePlayer(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                else:
                    for action in VALID_ACTIONS:
                        if event.key in ACTION_KEYS[action]:
                            self.player.move(action)
                            self.updateScore()
                            self.moveEnemies()
                            break

    def moveEnemies(self) -> None:
        for enemy in self.enemies:
            if action := enemy.chase(self.player):
                enemy.move(action)
            if enemy.pos == self.player.pos:
                self.running = False

    def main(self) -> None:
        while self.running:
            # Todo: how to updateScore here rather than in movePlayer?
            # self.updateScore()
            self.drawBg()
            self.player.draw()
            for enemy in self.enemies:
                enemy.draw()
            self.movePlayer()
            # Todo: how to moveEnemies here rather than in movePlayer?
            # self.moveEnemies()
            pygame.display.update()
        pygame.quit()
