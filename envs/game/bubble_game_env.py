import gym
from gym import spaces
import numpy as np
import pygame
import math

# Player class
class Player:
    def __init__(self):
        self.width = 30
        self.height = 50
        self.x = 400
        self.y = 550
        self.speed = 4
        self.color = (0, 0, 255)

    def move(self, keys):
        if keys[pygame.K_LEFT]:
            self.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.x += self.speed
        self.x = max(0, min(self.x, 800 - self.width))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

    def get_center(self):
        return self.x + self.width // 2, self.y

# Bullet class
class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 6
        self.speed = 10
        self.active = True
        self.color = (255, 0, 0)

    def update(self):
        self.y -= self.speed
        if self.y < 0:
            self.active = False

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

# Bubble class
class Bubble:
    def __init__(self, x, y, size, x_vel, y_vel):
        self.x = x
        self.y = y
        self.size = size
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.gravity = 0.3
        self.color = (0, 255, 0)

    def update(self):
        self.x += self.x_vel
        self.y += self.y_vel
        if self.x - self.size < 0 or self.x + self.size > 800:
            self.x_vel *= -1
        if self.y + self.size > 600:
            self.y = 600 - self.size
            self.y_vel = -abs(self.y_vel)
        else:
            self.y_vel += self.gravity
        if self.y - self.size < 0:
            self.y = self.size
            self.y_vel = abs(self.y_vel)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

# Custom Game Environment (Gym)
class BubbleGameEnv(gym.Env):
    def __init__(self):
        super(BubbleGameEnv, self).__init__()

        # Action space: move left, right, shoot
        self.action_space = spaces.Discrete(3)

        # Observation space: screen size (RGB)
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8)

        # Initialize game
        self.player = Player()
        self.bullet = None
        self.bubbles = [Bubble(100, 100, 40, 2, -3), Bubble(500, 150, 40, -1, -2)]

        # Initialize pygame screen
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Bubble Trouble - DRL")

    def reset(self):
        # Reset the game state
        self.player = Player()
        self.bullet = None
        self.bubbles = [Bubble(100, 100, 40, 2, -3), Bubble(500, 150, 40, -1, -2)]
        return self.render(mode="rgb_array")

    def step(self, action):
        # Execute the chosen action
        if action == 0:  # move left
            self.player.x -= self.player.speed
        elif action == 1:  # move right
            self.player.x += self.player.speed
        elif action == 2:  # shoot
            if self.bullet is None:
                bullet_x, bullet_y = self.player.get_center()
                self.bullet = Bullet(bullet_x, bullet_y)

        # Update bullet if it exists
        if self.bullet:
            self.bullet.update()
            if self.bullet.y < 0:
                self.bullet = None

        # Update bubbles
        for bubble in self.bubbles[:]:
            bubble.update()

            # Collision detection (with bullet or player)
            if self.bullet:
                if self.bullet.x > bubble.x - bubble.size and self.bullet.x < bubble.x + bubble.size:
                    if self.bullet.y > bubble.y - bubble.size and self.bullet.y < bubble.y + bubble.size:
                        if bubble.size > 15:
                            new_size = bubble.size // 2
                            self.bubbles.append(Bubble(bubble.x, bubble.y, new_size, -abs(bubble.x_vel), -8))
                            self.bubbles.append(Bubble(bubble.x, bubble.y, new_size, abs(bubble.x_vel), -8))
                        self.bubbles.remove(bubble)
                        self.bullet = None
                        break

            if (bubble.x - bubble.size < self.player.x + self.player.width and
                bubble.x + bubble.size > self.player.x and
                bubble.y - bubble.size < self.player.y + self.player.height and
                bubble.y + bubble.size > self.player.y):
                return self.render(mode='rgb_array'), -100, True, {}

        # Reward for completing level (all bubbles popped)
        if len(self.bubbles) == 0:
            self.bubbles = [Bubble(100, 100, 40, 2, -3), Bubble(500, 150, 40, -1, -2)]
            return self.render(mode="rgb_array"), 10, False, {}

        # Default step return
        return self.render(mode='rgb_array'), 0, False, {}

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))  # Clear the screen
        self.player.draw(self.screen)

        for bubble in self.bubbles:
            bubble.draw(self.screen)

        if self.bullet:
            self.bullet.draw(self.screen)

        if mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen)
        elif mode == "human":
            pygame.display.flip()

    def close(self):
        pygame.quit()
