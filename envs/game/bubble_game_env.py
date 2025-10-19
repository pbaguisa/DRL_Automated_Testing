import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import numpy as np
import pygame
import math
import random  # private RNG seeded per env

# Initialize Pygame (safe even headless)
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
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
        # body
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        # eyes + smile (cute face)
        head_cx = self.x + self.width // 2
        head_cy = self.y + self.height // 3
        eye_r   = max(2, self.width // 10)
        eye_dx  = self.width // 4
        eye_y   = head_cy - self.height // 6
        pygame.draw.circle(screen, WHITE, (head_cx - eye_dx, eye_y), eye_r)
        pygame.draw.circle(screen, WHITE, (head_cx + eye_dx, eye_y), eye_r)
        smile_rect = pygame.Rect(head_cx - self.width // 4, head_cy, self.width // 2, self.height // 3)
        pygame.draw.arc(screen, WHITE, smile_rect, math.pi/8, math.pi - math.pi/8, 2)

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

    def collide_with_player(self, player: Player) -> bool:
        player_left, player_right, player_top, player_bottom = player.get_edges()
        dist_x = abs(self.x - (player_left + player_right) / 2)
        dist_y = abs(self.y - (player_top + player_bottom) / 2)
        if dist_x > (player_right - player_left) / 2 + self.size:
            return False
        if dist_y > (player_bottom - player_top) / 2 + self.size:
            return False
        return True

    def collide_with_point(self, point_x, point_y) -> bool:
        dist = math.hypot(self.x - point_x, self.y - point_y)
        return dist < self.size


class BubbleGameEnv(gym.Env):
    """
    Bubble Trouble–style env with two reward modes:
      - 'survivor'   : live long, position safely, pop when possible
      - 'speedrunner': end fast by popping quickly; time penalized
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None,
                 reward_mode: str = "survivor"):
        super().__init__()
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.render_mode = render_mode
        self.reward_mode = reward_mode  # "survivor" | "speedrunner"

        # Actions: 0=left, 1=right, 2=shoot, 3=no-op
        self.action_space = spaces.Discrete(4)

        # Observation: [player x,y, bullet x,y, bubble1 x,y, bubble2 x,y]
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.array([WIDTH, HEIGHT, WIDTH, HEIGHT, WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=np.float32),
            shape=(8,), dtype=np.float32
        )

        self.score = 0
        self.frames = 0
        self.max_steps = 2000  # reasonable cap

        # shooting reliability
        self._shoot_cooldown = 0
        self._shoot_cooldown_max = 2

        # --- episode metrics (for eval) ---
        self._shots = 0
        self._pops = 0
        self._wall_frames = 0
        self._frames_alive = 0
        self._dist_accum = 0.0

        # pygame lazy init
        self._pygame = None
        self._screen = None
        self._clock = None

    # ---------- helpers ----------
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

    def _end_info(self):
        # compute aggregate metrics at episode end
        avg_distance = (self._dist_accum / self._frames_alive) if self._frames_alive > 0 else 0.0
        wall_ratio = self._wall_frames / max(1, self._frames_alive)
        accuracy = (self._pops / self._shots) if self._shots > 0 else 0.0
        return {
            "reward_mode": self.reward_mode,
            "shots": int(self._shots),
            "pops": int(self._pops),
            "frames_alive": int(self._frames_alive),
            "wall_ratio": float(wall_ratio),
            "accuracy": float(accuracy),
            "avg_dist": float(avg_distance),
        }

    # ---------- gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rnd.seed(seed)
            self._np_rng = np.random.default_rng(seed)

        self._shoot_cooldown = 0

        # reset episode metrics
        self._shots = 0
        self._pops = 0
        self._wall_frames = 0
        self._frames_alive = 0
        self._dist_accum = 0.0

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

        info = {"score": self.score, "bubbles_active": len(self.bubbles), "reward_mode": self.reward_mode}
        return self._obs(), info

    def step(self, action):
        # Coerce to int robustly
        if isinstance(action, np.ndarray):
            action = int(action.item() if action.ndim == 0 else action[0])
        elif isinstance(action, (list, tuple)):
            action = int(action[0])
        else:
            action = int(action)

        total_reward = 0.0
        popped_this_step = False

        # cooldown
        if self._shoot_cooldown > 0:
            self._shoot_cooldown -= 1

        # actions
        if action == 0:
            self.player.x -= self.player.speed
        elif action == 1:
            self.player.x += self.player.speed
        elif action == 2:
            if self.bullet is None and self._shoot_cooldown == 0:
                bx, by = self.player.get_center()
                self.bullet = Bullet(bx, by)
                self._shoot_cooldown = self._shoot_cooldown_max
                self._shots += 1
                # mode-aware shoot reward
                if self.reward_mode == "survivor":
                    total_reward += 0.5
                elif self.reward_mode == "speedrunner":
                    total_reward += 0.75
        elif action == 3:
            pass  # no-op

        # bounds
        self.player.x = max(0, min(self.player.x, WIDTH - self.player.width))

        # metrics: frames + wall time
        self._frames_alive += 1
        if self.player.x <= 0 or self.player.x >= (WIDTH - self.player.width):
            self._wall_frames += 1

        # bullet update
        if self.bullet:
            self.bullet.update()
            if not self.bullet.active:
                self.bullet = None

        # bubbles + collisions
        px_center = float(self.player.x + self.player.width * 0.5)
        for bubble in self.bubbles[:]:
            bubble.update()

            # accumulate average horizontal distance to nearest bubble
            # (we'll pick nearest again below for shaping, but this keeps cost O(n))
            # do it here by comparing to current bubble; we'll keep the min
            # simpler: just compute after loop using min(); below we also do shaping with nearest

            # player collision => terminate
            if bubble.collide_with_player(self.player):
                info = self._end_info() | {"deaths": 1}
                return self._obs(), -100.0, True, False, info

            # bullet hit => split/pop
            if self.bullet and bubble.collide_with_point(self.bullet.x, self.bullet.y):
                if bubble.size > 15:
                    new_size = bubble.size // 2
                    self.bubbles.append(Bubble(bubble.x, bubble.y, new_size, -abs(bubble.x_vel), -8))
                    self.bubbles.append(Bubble(bubble.x, bubble.y, new_size,  abs(bubble.x_vel), -8))
                self.bubbles.remove(bubble)
                self.bullet = None
                self._pops += 1
                popped_this_step = True
                # (pop reward added in mode-specific section)
                continue

        # ----- mode-specific shaping -----
        # nearest bubble helpers
        if self.bubbles:
            nearest = min(self.bubbles, key=lambda b: abs(b.x - px_center))
            dx = float(abs(nearest.x - px_center))
            safe_under = (self.player.y + self.player.height) <= (nearest.y - nearest.size - 10)
            # accumulate distance for avg_dist metric
            self._dist_accum += dx
        else:
            nearest, dx, safe_under = None, WIDTH, False

        at_left_wall  = self.player.x <= 0.0
        at_right_wall = self.player.x >= (WIDTH - self.player.width)

        if self.reward_mode == "survivor":
            ALIVE_BONUS = 0.05
            ALIGN_GAIN  = 0.002
            SAFE_BONUS  = 0.1
            WALL_PEN    = 0.05
            POP_REWARD  = 10.0
            BULLET_DRAG = 0.005

            total_reward += ALIVE_BONUS

            if nearest is not None:
                total_reward += ALIGN_GAIN * (WIDTH - min(dx, WIDTH))
                if safe_under and dx < (nearest.size + self.player.width * 0.5):
                    total_reward += SAFE_BONUS

            if at_left_wall or at_right_wall:
                total_reward -= WALL_PEN

            if self.bullet:
                total_reward -= BULLET_DRAG

            if popped_this_step:
                total_reward += POP_REWARD

        elif self.reward_mode == "speedrunner":
            STEP_PENALTY = 0.01
            ALIGN_GAIN   = 0.004
            CLOSE_BONUS  = 0.15
            POP_REWARD   = 20.0
            # no bullet drag; no wall penalty

            total_reward -= STEP_PENALTY

            if nearest is not None:
                total_reward += ALIGN_GAIN * (WIDTH - min(dx, WIDTH))
                if safe_under and dx < (nearest.size + self.player.width * 0.5):
                    total_reward += CLOSE_BONUS

            if popped_this_step:
                total_reward += POP_REWARD
        else:
            # fallback minimal alive reward
            total_reward += 0.05

        # time limit
        truncated = False
        self.frames += 1
        if self.frames >= self.max_steps:
            truncated = True
            info = self._end_info() | {"deaths": 0}
            return self._obs(), total_reward, False, True, info

        # continue episode
        return self._obs(), total_reward, False, False, {}

    # ---------- rendering ----------
    def render(self, mode='human'):
        self._lazy_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        self._screen.fill(BLACK)
        self.player.draw(self._screen)
        for bubble in self.bubbles:
            bubble.draw(self._screen)
        if self.bullet:
            self.bullet.draw(self._screen)

        if mode == 'rgb_array':
            img = pygame.surfarray.array3d(self._screen)
            return img.transpose(2, 0, 1)
        elif mode == 'human':
            pygame.display.flip()
            if self._clock:
                self._clock.tick(self.metadata.get("render_fps", 60))

    def close(self):
        if self._pygame:
            pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None

    def _lazy_pygame(self):
        if self._pygame is None:
            import pygame as _pg
            self._pygame = _pg
            self._pygame.init()
            self._screen = _pg.display.set_mode((WIDTH, HEIGHT))
            self._clock = _pg.time.Clock()
