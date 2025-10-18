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
        self._shoot_cooldown = 0
        self._shoot_cooldown_max = 2

    def move(self, keys):
        if keys[pygame.K_LEFT]:
            self.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.x += self.speed
        self.x = max(0, min(self.x, WIDTH - self.width))

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

        # Head center (roughly the top half of the rectangle)
        head_center_x = self.x + self.width // 2
        head_center_y = self.y + self.height // 3
        head_radius = self.width // 2

        # Eyes 
        eye_radius = max(2, self.width // 10)
        eye_offset_x = self.width // 4
        eye_y = head_center_y - self.height // 6
        pygame.draw.circle(screen, WHITE, (head_center_x - eye_offset_x, eye_y), eye_radius)
        pygame.draw.circle(screen, WHITE, (head_center_x + eye_offset_x, eye_y), eye_radius)

        # Frown
        smile_rect = pygame.Rect(
            head_center_x - self.width // 4,
            head_center_y,
            self.width // 2,
            self.height // 2
        )
        pygame.draw.arc(screen, WHITE, smile_rect, math.pi / 8, math.pi - math.pi / 8, 2)


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

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low = np.zeros(8, dtype=np.float32),
            high = np.array([WIDTH, HEIGHT, WIDTH, HEIGHT, WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=np.float32),
            shape = (8,), dtype=np.float32
        )
        self.score = 0
        self.frames = 0
        self.max_steps = 2000

        self._pygame = None
        self._screen = None
        self._clock = None

        # defer actual state creation to reset()

    def _obs(self):
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

        self._shoot_cooldown = 0
        self._shoot_cooldown_max = 2

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
        # Coerce SB3 / numpy actions to a clean Python int
        if isinstance(action, np.ndarray):
            action = int(action.item() if action.ndim == 0 else action[0])
        elif isinstance(action, (list, tuple)):
            action = int(action[0])
        else:
            action = int(action)

        total_reward = 0.0

        # Tick cooldown
        if self._shoot_cooldown > 0:
            self._shoot_cooldown -= 1

        # Actions
        if action == 0:              # Move left
            self.player.x -= self.player.speed
        elif action == 1:            # Move right
            self.player.x += self.player.speed
        elif action == 2:            # Shoot
            if self.bullet is None and self._shoot_cooldown == 0:
                bx, by = self.player.get_center()
                self.bullet = Bullet(bx, by)
                self._shoot_cooldown = self._shoot_cooldown_max
                total_reward += 0.1   # small reward just for attempting to shoot
        elif action == 3:
            pass

        # Keep player in bounds
        self.player.x = max(0, min(self.player.x, WIDTH - self.player.width))

        # Update bullet
        if self.bullet:
            self.bullet.update()
            if not self.bullet.active:
                self.bullet = None
            else:
                # small penalty while a bullet is in-flight (encourages frequent shots)
                total_reward -= 0.01

        terminated = False

        # Update bubbles and compute rewards/collisions
        for bubble in self.bubbles[:]:
            bubble.update()

            # # Dense reward for being safely under the bubble when horizontally near
            # horiz_overlap = abs((self.player.x + self.player.width / 2) - bubble.x) < (self.player.width / 2 + bubble.size)
            # safe_under = (self.player.y + self.player.height) <= (bubble.y - bubble.size - 10)
            # if horiz_overlap and safe_under:
            #     total_reward += 0.5

            # Neatest bubble (horizontal distance)
            px_center = float(self.player.x + self.player.width * 0.5)

            if self.bubbles:
                nearest = min(self.bubbles, key=lambda b: abs(b.x - px_center))

                # Reward grows higher when player directly aligns; capped so its not aimbot lol
                dx = float(abs(nearest.x - px_center))
                total_reward += 0.002 * (WIDTH - min(dx, WIDTH))

                # Reward for passing underneath without collision
                safe_under = (self.player.y + self.player.height) <= (nearest.y - nearest.size - 10)
                if safe_under and dx < (nearest.size + self.player.width * 0.5):
                    total_reward += 0.1

            # Wall camping nerf, force movement (penalty)
            at_left_wall = self.player.x <= 0
            at_right_wall = self.player.x >= (WIDTH - self.player.width)
            if at_left_wall or at_right_wall:
                total_reward -= 0.05

            # Player collision -> end episode
            if bubble.collide_with_player(self.player):
                return self._obs(), -100.0, True, False, {}

            # Bullet hits bubble -> pop/split
            if self.bullet and bubble.collide_with_point(self.bullet.x, self.bullet.y):
                if bubble.size > 15:
                    new_size = bubble.size // 2
                    self.bubbles.append(Bubble(bubble.x, bubble.y, new_size, -abs(bubble.x_vel), -8))
                    self.bubbles.append(Bubble(bubble.x, bubble.y, new_size,  abs(bubble.x_vel), -8))
                self.bubbles.remove(bubble)
                self.bullet = None
                total_reward += 10.0
                # continue to process others after mutation
                continue

        # Time-limit truncation
        truncated = False
        self.frames += 1
        if self.frames >= self.max_steps:
            truncated = True

        return self._obs(), total_reward, terminated, truncated, {}


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