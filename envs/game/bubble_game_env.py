# envs/game/bubble_game_env.py
import os
import math
import random
from typing import Optional, Tuple, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# -----------------------------
# Minimal YAML loader
# -----------------------------
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None

def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    if yaml is None:
        raise ImportError(
            f"Config file found at {p} but PyYAML is not installed. "
            "Install with: pip install pyyaml"
        )
    with p.open("r") as f:
        data = yaml.safe_load(f)
    return data or {}

# -----------------------------
# Pygame setup
# -----------------------------
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)


# -----------------------------
# Entities 
# -----------------------------
class Player:
    def __init__(self, x: int):
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
        # simple face
        head_cx = self.x + self.width // 2
        head_cy = self.y + self.height // 3
        eye_r   = max(2, self.width // 10)
        eye_dx  = self.width // 4
        eye_y   = head_cy - self.height // 6
        pygame.draw.circle(screen, WHITE, (head_cx - eye_dx, eye_y), eye_r)
        pygame.draw.circle(screen, WHITE, (head_cx + eye_dx, eye_y), eye_r)
        smile_rect = pygame.Rect(head_cx - self.width // 4, head_cy, self.width // 2, self.height // 3)
        pygame.draw.arc(screen, WHITE, smile_rect, math.pi/8, math.pi - math.pi/8, 2)

    def get_center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y)

    def get_edges(self) -> Tuple[int, int, int, int]:
        left = self.x
        right = self.x + self.width
        top = self.y
        bottom = self.y + self.height
        return left, right, top, bottom


class Bullet:
    def __init__(self, x: int, y: int):
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
    def __init__(self, x: float, y: float, size: int, x_vel: float, y_vel: float):
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

        # bounce on walls
        if self.x - self.size < 0 or self.x + self.size > WIDTH:
            self.x_vel *= -1

        # floor/ceiling with gravity
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

    def collide_with_point(self, point_x: float, point_y: float) -> bool:
        dist = math.hypot(self.x - point_x, self.y - point_y)
        return dist < self.size


# -----------------------------
# Environment
# -----------------------------
class BubbleGameEnv(gym.Env):
    """
    A small bubble shooter environment.

    reward_mode:
      - 'survivor'    : live long, position safely
      - 'speedrunner' : finish fast; time costs reward; pops are highly valued
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        reward_mode: str = "survivor",
    ):
        super().__init__()

        # RNG
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self.render_mode = render_mode
        self.reward_mode = reward_mode

        # -----------------------------
        # Load reward shaping from configs/rewards/<mode>.yaml
        # (Fallbacks preserve existing behavior if files are absent)
        # -----------------------------
        cfg_path = os.path.join("configs", "rewards", f"{self.reward_mode}.yaml")
        reward_cfg = _load_yaml(cfg_path)

        # shared
        self._shoot_reward = float(reward_cfg.get("shoot_reward",
                                   0.5 if self.reward_mode == "survivor" else 0.75))

        # survivor defaults
        self._sv_alive_bonus = float(reward_cfg.get("alive_bonus", 0.05))
        self._sv_align_gain  = float(reward_cfg.get("align_gain", 0.002))
        self._sv_safe_bonus  = float(reward_cfg.get("safe_bonus", 0.1))
        self._sv_wall_pen    = float(reward_cfg.get("wall_penalty", 0.05))
        self._sv_pop_reward  = float(reward_cfg.get("pop_reward", 10.0))
        self._sv_bullet_drag = float(reward_cfg.get("bullet_drag", 0.005))

        # speedrunner defaults
        self._sr_step_penalty = float(reward_cfg.get("step_penalty", 0.01))
        self._sr_align_gain   = float(reward_cfg.get("align_gain", 0.004))
        self._sr_close_bonus  = float(reward_cfg.get("close_bonus", 0.15))
        self._sr_pop_reward   = float(reward_cfg.get("pop_reward", 20.0))

        # ---- bubble splitting ----
        self._min_split_size = 12  # when size <= this, bubble is destroyed instead of splitting

        # -----------------------------
        # Action & Observation spaces
        # -----------------------------
        # 0=left, 1=right, 2=shoot, 3=no-op
        self.action_space = spaces.Discrete(4)

        # obs: [player x,y, bullet x,y, bubble1 x,y, bubble2 x,y]
        self.observation_space = spaces.Box(
            low=np.zeros(8, dtype=np.float32),
            high=np.array([WIDTH, HEIGHT, WIDTH, HEIGHT, WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=np.float32),
            shape=(8,),
            dtype=np.float32,
        )

        # episode state / metrics
        self.frames = 0
        self.max_steps = 200000  # large cap

        self._shoot_cooldown = 0
        self._shoot_cooldown_max = 2

        self._shots = 0
        self._pops = 0
        self._wall_frames = 0
        self._frames_alive = 0
        self._dist_accum = 0.0

        # pygame stuff (lazy init)
        self._pygame = None
        self._screen = None
        self._clock = None

        # game objects
        self.player: Optional[Player] = None
        self.bullet: Optional[Bullet] = None
        self.bubbles: List[Bubble] = []

        self.reset(seed=seed)

    # -----------------------------
    # Gym API
    # -----------------------------
    def seed(self, seed: Optional[int] = None):
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _spawn_bubble(self) -> Bubble:
        size = self._rnd.choice([22, 26, 30, 34])
        x = self._rnd.randint(size + 5, WIDTH - size - 5)
        y = self._rnd.randint(size + 5, HEIGHT // 3)
        x_vel = self._rnd.choice([-2.2, -1.8, 1.8, 2.2])
        y_vel = self._rnd.choice([-3.5, -2.8, -3.0])
        return Bubble(x, y, size, x_vel, y_vel)

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)

        self.frames = 0
        self._shoot_cooldown = 0

        self._shots = 0
        self._pops = 0
        self._wall_frames = 0
        self._frames_alive = 0
        self._dist_accum = 0.0

        # start with two bubbles
        self.player = Player(x=WIDTH // 2 - 15)
        self.bullet = None
        self.bubbles = [self._spawn_bubble(), self._spawn_bubble()]

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        self.frames += 1
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # shooting cooldown
        if self._shoot_cooldown > 0:
            self._shoot_cooldown -= 1

        # move by action
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
                total_reward += self._shoot_reward

        # clamp player
        self.player.x = max(0, min(self.player.x, WIDTH - self.player.width))

        if self.bullet is not None:
            self.bullet.update()
            if not self.bullet.active:
                self.bullet = None

        for b in self.bubbles:
            b.update()

        # collisions: player vs bubble
        for b in list(self.bubbles):
            if b.collide_with_player(self.player):
                terminated = True
                break

        # ---- collisions: bullet vs bubble  ----
        popped_this_step = False
        if self.bullet is not None:
            hit_index = None
            for i, b in enumerate(self.bubbles):
                if b.collide_with_point(self.bullet.x, self.bullet.y):
                    hit_index = i
                    break

            if hit_index is not None:
                b = self.bubbles[hit_index]
                self.bullet = None
                popped_this_step = True
                self._pops += 1

                # split at impact into TWO smaller bubbles at same location
                if b.size > self._min_split_size:
                    child_size = max(self._min_split_size, b.size // 2)
                    # give children mirrored horizontal velocities and upward kick
                    vx_mag = max(1.5, abs(b.x_vel))
                    vy_up = -abs(max(2.5, abs(b.y_vel)))
                    left_child  = Bubble(b.x, b.y, child_size, -vx_mag, vy_up)
                    right_child = Bubble(b.x, b.y, child_size, +vx_mag, vy_up)
                    # replace hit bubble with left child, append right child
                    self.bubbles[hit_index] = left_child
                    self.bubbles.insert(hit_index + 1, right_child)
                else:
                    # too small to split: remove it entirely
                    del self.bubbles[hit_index]
        # -------------------------------------------------------------

        # reward shaping helpers
        px_center = self.player.x + self.player.width / 2
        nearest, dx = None, None
        if self.bubbles:
            nearest = min(self.bubbles, key=lambda b: abs(b.x - px_center))
            dx = abs(nearest.x - px_center)

        safe_under = nearest is not None and nearest.y + nearest.size < self.player.y

        at_left_wall = self.player.x <= 0
        at_right_wall = self.player.x >= WIDTH - self.player.width
        if at_left_wall or at_right_wall:
            self._wall_frames += 1

        self._frames_alive += 1
        if dx is not None:
            self._dist_accum += dx

        if self.reward_mode == "survivor":
            ALIVE_BONUS = self._sv_alive_bonus
            ALIGN_GAIN  = self._sv_align_gain
            SAFE_BONUS  = self._sv_safe_bonus
            WALL_PEN    = self._sv_wall_pen
            POP_REWARD  = self._sv_pop_reward
            BULLET_DRAG = self._sv_bullet_drag

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
            STEP_PENALTY = self._sr_step_penalty
            ALIGN_GAIN   = self._sr_align_gain
            CLOSE_BONUS  = self._sr_close_bonus
            POP_REWARD   = self._sr_pop_reward

            total_reward -= STEP_PENALTY

            if nearest is not None:
                total_reward += ALIGN_GAIN * (WIDTH - min(dx, WIDTH))
                if safe_under and dx < (nearest.size + self.player.width * 0.5):
                    total_reward += CLOSE_BONUS

            if popped_this_step:
                total_reward += POP_REWARD
        else:
            total_reward += 0.05  # fallback

        # truncation
        if self.frames >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        return obs, float(total_reward), terminated, truncated, info

    # -----------------------------
    # Helpers
    # -----------------------------
    def _get_obs(self) -> np.ndarray:
        # player
        px = float(self.player.x)
        py = float(self.player.y)

        # bullet
        if self.bullet is None:
            bx, by = 0.0, 0.0
        else:
            bx, by = float(self.bullet.x), float(self.bullet.y)

        # export first two bubbles; pad with zeros if fewer
        coords: List[Tuple[float, float]] = [(float(b.x), float(b.y)) for b in self.bubbles[:2]]
        while len(coords) < 2:
            coords.append((0.0, 0.0))

        obs = np.array([px, py, bx, by, coords[0][0], coords[0][1], coords[1][0], coords[1][1]], dtype=np.float32)
        return obs

    # -----------------------------
    # Rendering
    # -----------------------------
    def render(self, mode: str = "human"):
        if mode not in self.metadata["render_modes"]:
            mode = "human"

        if self._pygame is None:
            self._lazy_pygame()

        # handle window close
        for event in self._pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._screen.fill(BLACK)

        # floor line (visual reference)
        pygame.draw.line(self._screen, WHITE, (0, HEIGHT - 1), (WIDTH, HEIGHT - 1), 1)

        # draw entities
        self.player.draw(self._screen)
        for b in self.bubbles:
            b.draw(self._screen)
        if self.bullet:
            self.bullet.draw(self._screen)

        if mode == "human":
            self._pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            arr = self._pygame.surfarray.array3d(self._screen)
            return np.transpose(arr, (1, 0, 2))

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
