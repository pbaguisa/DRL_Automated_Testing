import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple
import numpy as np
import pygame
import math
import random  # To randomize player and bubble positions

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

class Player:
    def __init__(self, x):
        self.width = 30
        self.height = 50
        self.x = x
        self.y = HEIGHT - self.height - 10
        self.speed = 4
        self.color = BLUE

    def move(self, keys):
        if keys[pygame.K_LEFT]:
            self.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.x += self.speed
        self.x = max(0, min(self.x, WIDTH - self.width))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

    def get_center(self):
        return (self.x + self.width // 2, self.y)

    def get_edges(self):
        left = self.x
        right = self.x + self.width
        top = self.y
        bottom = self.y + self.height
        return left, right, top, bottom

class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 6
        self.speed = 10
        self.color = RED
        self.active = True

    def update(self):
        self.y -= self.speed
        if self.y < 0:
            self.active = False

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)

class Bubble:
    def __init__(self, x, y, size, x_vel, y_vel):
        self.x = x
        self.y = y
        self.size = size
        self.color = GREEN
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.gravity = 0.1

    def update(self):
        self.x += self.x_vel
        self.y += self.y_vel
        if self.x - self.size < 0 or self.x + self.size > WIDTH:
            self.x_vel *= -1
        if self.y + self.size > HEIGHT:
            self.y = HEIGHT - self.size
            self.y_vel = -abs(self.y_vel)
        else:
            self.y_vel += self.gravity

        if self.y - self.size < 0:
            self.y = self.size
            self.y_vel = abs(self.y_vel)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

    def collide_with_player(self, player):
        player_left, player_right, player_top, player_bottom = player.get_edges()
        dist_x = abs(self.x - (player_left + player_right) / 2)
        dist_y = abs(self.y - (player_top + player_bottom) / 2)
        if dist_x > (player_right - player_left) / 2 + self.size:
            return False
        if dist_y > (player_bottom - player_top) / 2 + self.size:
            return False
        return True
    
    def collide_with_point(self, point_x, point_y):
        dist = math.hypot(self.x - point_x, self.y - point_y)
        return dist < self.size

class BubbleGameEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.array([WIDTH, HEIGHT, WIDTH, HEIGHT, WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=np.float32),
            shape=(8,), dtype=np.float32
        )
        self.score = 0
        self.frames = 0
        self.max_steps = 100000

        self._pygame = None
        self._screen = None
        self._clock = None

        # defer actual state creation to reset()

    def _obs(self) -> np.ndarray:
        bx = [0.0, 0.0]; by = [0.0, 0.0]
        n = min(2, len(self.bubbles))
        for i in range(n):
            bx[i] = float(self.bubbles[i].x); by[i] = float(self.bubbles[i].y)
        obs = np.array([
            float(self.player.x), float(self.player.y),
            float(self.bullet.x if self.bullet else 0.0),
            float(self.bullet.y if self.bullet else 0.0),
            bx[0], by[0], bx[1], by[1]
        ], dtype=np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rnd.seed(seed)
            self._np_rng = np.random.default_rng(seed)

        # randomized state via private RNG
        self.player = Player(x=self._rnd.randint(0, WIDTH - 30))
        self.bullet = None
        self.bubbles = [
            Bubble(self._rnd.randint(0, WIDTH), self._rnd.randint(100, HEIGHT - 100), 40,
                   self._rnd.uniform(-2, 2), self._rnd.uniform(-3, -1)),
            Bubble(self._rnd.randint(0, WIDTH), self._rnd.randint(100, HEIGHT - 100), 40,
                   self._rnd.uniform(-2, 2), self._rnd.uniform(-3, -1))
        ]
        self.frames = 0

        info = {"score": self.score, "bubbles_active": len(self.bubbles)}
        return self._obs(), info


    def step(self, action):
        # Process action (move left, move right, shoot bullet)


        if action == 0:  # Move left
            self.player.x -= self.player.speed
        elif action == 1:  # Move right
            self.player.x += self.player.speed
        elif action == 2:  # Shoot bullet
            if self.bullet is None:  # Only shoot if there is no bullet already
                bullet_x, bullet_y = self.player.get_center()
                self.bullet = Bullet(bullet_x, bullet_y)

        # Prevent player from moving out of bounds
        self.player.x = max(0, min(self.player.x, WIDTH - self.player.width))

        # Update bullet if it exists
        if self.bullet:
            self.bullet.update()
            if self.bullet.y < 0:  # If the bullet goes off-screen (top side), remove it
                self.bullet = None

        terminated = False
        total_reward = 0  # Initialize reward for this step

        for bubble in self.bubbles[:]:
            bubble.update()
            
            # reward for being horizontally close under the bubble while not colliding
            horiz_overlap = abs((self.player.x + self.player.width/2) - bubble.x) < (self.player.width/2 + bubble.size)
            safe_under = (self.player.y + self.player.height) <= (bubble.y - bubble.size - 10)  # 10px safety margin
            if horiz_overlap and safe_under:
                total_reward += 0.1  # small dense reward

        
            if bubble.collide_with_player(self.player):
                terminated = True  # Episode ends if player collides with a bubble
                return np.array([self.player.x, self.player.y, self.bullet.x if self.bullet else 0,
                                 self.bullet.y if self.bullet else 0, self.bubbles[0].x, self.bubbles[0].y,
                                 self.bubbles[1].x, self.bubbles[1].y]), -100, terminated, False, {}

            # Reward for shooting the bubble
            if self.bullet and bubble.collide_with_point(self.bullet.x, self.bullet.y):
                if bubble.size > 15:
                    new_size = bubble.size // 2
                    self.bubbles.append(Bubble(bubble.x, bubble.y, new_size, -abs(bubble.x_vel), -8))
                    self.bubbles.append(Bubble(bubble.x, bubble.y, new_size, abs(bubble.x_vel), -8))
                self.bubbles.remove(bubble)  # Remove the bubble
                self.bullet = None  # Destroy the bullet
                total_reward += 10  # Reward for shooting a bubble

        truncated = False
        self.frames += 1  # Increment frames counter each time step is taken
        if self.frames >= self.max_steps:
            truncated = True  # Mark the episode as truncated if it exceeds max steps

        # Return the 5 values: observation, reward, terminated, truncated, info
        return np.array([self.player.x, self.player.y, self.bullet.x if self.bullet else 0,
                         self.bullet.y if self.bullet else 0, self.bubbles[0].x, self.bubbles[0].y,
                         self.bubbles[1].x, self.bubbles[1].y]), total_reward, terminated, truncated, {}

    def render(self, mode='human'):
        self._lazy_pygame()  # Ensure Pygame and screen are initialized

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        self._screen.fill((0, 0, 0))  # Fill the screen with black (clear previous frame)
        self.player.draw(self._screen)
        for bubble in self.bubbles:
            bubble.draw(self._screen)
        if self.bullet:
            self.bullet.draw(self._screen)

        if mode == 'rgb_array':
            img = pygame.surfarray.array3d(self._screen)
            return img.transpose(2, 0, 1)  # Convert to (C, H, W) format for ML
        elif mode == 'human':
            pygame.display.flip()  # Update the display for human visualization

    def _lazy_pygame(self):
        if self._pygame is None:
            import pygame
            self._pygame = pygame
            self._pygame.init()
            self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self._clock = pygame.time.Clock()