import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict
import numpy as np
import pygame
import os
import random
import yaml

# Constants
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 300
TILE_SIZE = 16
FPS = 60
GRAVITY = 0.5
MAX_FALL_SPEED = 10
WALK_SPEED = 2
SPRINT_SPEED = 4
WALK_ACCEL = 0.5
SPRINT_ACCEL = 0.8
FRICTION = 0.8
JUMP_IMPULSE = -9
COYOTE_TIME = 3
JUMP_BUFFER = 3
WORLD_WIDTH = 3392  # ~212 tiles

# Colors
SKY_BLUE = (92, 148, 252)
GROUND_BROWN = (132, 94, 60)
BRICK_RED = (200, 76, 12)
QUESTION_YELLOW = (252, 188, 0)
PIPE_GREEN = (60, 188, 12)
COIN_GOLD = (252, 224, 56)
GOOMBA_BROWN = (164, 100, 34)
LUIGI_RED = (252, 60, 60)
LUIGI_BLUE = (0, 0, 252)
FLAG_WHITE = (252, 252, 252)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Don't initialize pygame globally - will be done lazily in the environment
# pygame.init()
# screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# pygame.display.set_caption("Super Luigi Bros")
# clock = pygame.time.Clock()
# font = pygame.font.Font(None, 16)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Sprite globals - will be loaded lazily
USE_SPRITES = False
luigi_small_img = None
luigi_big_img = None
goomba_img = None
mushroom_img = None

class Camera:
    def __init__(self):
        self.x = 0
        self.max_x = 0
        
    def update(self, target_x):
        # Camera follows player with lead
        target_cam_x = target_x - SCREEN_WIDTH // 3
        self.x += (target_cam_x - self.x) * 0.1
        self.x = max(self.max_x, min(self.x, WORLD_WIDTH - SCREEN_WIDTH))
        self.max_x = max(self.max_x, self.x)
        
    def apply(self, x, y):
        return x - self.x, y

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.on_ground = False
        self.super = False
        self.width = 14  # Slightly thinner than a tile
        self.height = TILE_SIZE  # Start at half height (small Luigi)
        self.coyote_timer = 0
        self.jump_buffer = 0
        self.alive = True
        self.coins = 0  # Track coin count
        
    def get_rect(self):
        h = TILE_SIZE * 2 if self.super else TILE_SIZE
        return pygame.Rect(self.x, self.y, self.width, h)
        
    def update(self, keys, tiles, enemies, coins, powerups, camera):
        if not self.alive:
            return
            
        # Horizontal movement
        sprint = keys[pygame.K_LSHIFT]
        max_speed = SPRINT_SPEED if sprint else WALK_SPEED
        accel = SPRINT_ACCEL if sprint else WALK_ACCEL
        
        if keys[pygame.K_LEFT]:
            self.vx = max(self.vx - accel, -max_speed)
        elif keys[pygame.K_RIGHT]:
            self.vx = min(self.vx + accel, max_speed)
        else:
            self.vx *= FRICTION
            if abs(self.vx) < 0.1:
                self.vx = 0
                
        # Jump - Fixed height (no variable jump)
        if keys[pygame.K_SPACE]:
            self.jump_buffer = JUMP_BUFFER
        if self.jump_buffer > 0:
            self.jump_buffer -= 1
            if self.coyote_timer > 0:
                self.vy = JUMP_IMPULSE
                self.jump_buffer = 0
                self.coyote_timer = 0
                
        # Gravity
        self.vy = min(self.vy + GRAVITY, MAX_FALL_SPEED)
        
        # Horizontal collision
        self.x += self.vx
        rect = self.get_rect()
        for tile in tiles:
            if not (isinstance(tile, Brick) and tile.broken):
                if rect.colliderect(tile.rect):
                    if self.vx > 0:
                        self.x = tile.rect.left - self.width
                    elif self.vx < 0:
                        self.x = tile.rect.right
                    rect = self.get_rect()
                
        # Vertical collision
        self.y += self.vy
        rect = self.get_rect()
        self.on_ground = False
        hit_block = None  # Track the closest block above when jumping
        
        # Add forgiveness zone for hitting blocks - check slightly above player's head
        HIT_FORGIVENESS = 4  # Extra pixels above head to detect blocks
        
        for tile in tiles:
            if not (isinstance(tile, Brick) and tile.broken):
                if rect.colliderect(tile.rect):
                    if self.vy > 0:
                        self.y = tile.rect.top - rect.height
                        self.vy = 0
                        self.on_ground = True
                    elif self.vy < 0:
                        # When moving up, find the block directly above (lowest bottom)
                        if hit_block is None or tile.rect.bottom > hit_block.rect.bottom:
                            hit_block = tile
                        self.y = tile.rect.bottom
                        self.vy = 0
                    rect = self.get_rect()
                elif self.vy < 0:
                    # Check for blocks in the forgiveness zone above
                    forgiveness_rect = pygame.Rect(rect.left, rect.top - HIT_FORGIVENESS, rect.width, HIT_FORGIVENESS)
                    if forgiveness_rect.colliderect(tile.rect):
                        if hit_block is None or tile.rect.bottom > hit_block.rect.bottom:
                            hit_block = tile
        
        # Hit only the closest block above the player's head
        if hit_block is not None:
            hit_block.hit(self, powerups, coins)
                
        # Coyote time
        if self.on_ground:
            self.coyote_timer = COYOTE_TIME
        else:
            self.coyote_timer = max(0, self.coyote_timer - 1)
            
        # Enemy collision
        for enemy in enemies[:]:
            if rect.colliderect(enemy.get_rect()):
                # Check if stomping: player is falling and hitting from above
                enemy_rect = enemy.get_rect()
                # More forgiving stomp detection: check if player's bottom is in upper half of enemy
                # and player is moving downward
                stomp_threshold = enemy_rect.top + (enemy_rect.height * 0.6)
                if self.vy > 0 and rect.bottom <= stomp_threshold:
                    # Stomp
                    enemy.stomped(enemies)
                    self.vy = JUMP_IMPULSE * 0.5
                    # Skip damage check since we stomped successfully
                    continue
                else:
                    self.take_damage()
                
        # Powerup collection
        for powerup in powerups[:]:
            if rect.colliderect(powerup.get_rect()):
                if powerup.type == "mushroom":
                    if not self.super:
                        # Move player up when growing to prevent clipping into ground
                        self.y -= TILE_SIZE
                    self.super = True
                powerups.remove(powerup)
                
        # Death by falling
        if self.y > SCREEN_HEIGHT + 32:
            self.alive = False
            
        # Keep in bounds (world right boundary)
        self.x = min(self.x, WORLD_WIDTH - self.width)
        
        # Left screen boundary - player cannot go off left side of camera view
        if self.x < camera.x:
            self.x = camera.x
            self.vx = max(0, self.vx)  # Stop leftward movement
        
    def take_damage(self):
        if self.super:
            self.super = False
        else:
            self.alive = False
            
    def draw(self, screen, camera):
        rect = self.get_rect()
        x, y = camera.apply(rect.x, rect.y)
        
        if USE_SPRITES:
            # Use sprite images - scale up 1.3x for better visibility
            img = luigi_big_img if self.super else luigi_small_img
            scaled_width = int(rect.width * 1.3)
            scaled_height = int(rect.height * 1.3)
            scaled_img = pygame.transform.scale(img, (scaled_width, scaled_height))
            # Center horizontally, align to bottom of hitbox
            offset_x = (scaled_width - rect.width) // 2
            offset_y = scaled_height - rect.height
            screen.blit(scaled_img, (x - offset_x, y - offset_y))
        else:
            # Fallback to rectangles
            color = LUIGI_RED if not self.super else LUIGI_BLUE
            pygame.draw.rect(screen, color, (x, y, rect.width, rect.height))
            # Simple face (centered for 14px width)
            pygame.draw.circle(screen, BLACK, (int(x + 4), int(y + 6)), 2)
            pygame.draw.circle(screen, BLACK, (int(x + 10), int(y + 6)), 2)

class Tile:
    def __init__(self, x, y, tile_type):
        self.x = x
        self.y = y
        self.type = tile_type
        self.rect = pygame.Rect(x, y, TILE_SIZE, TILE_SIZE)
        
    def hit(self, player, powerups, coins):
        pass
        
    def draw(self, screen, camera):
        x, y = camera.apply(self.x, self.y)
        if self.type == "ground":
            pygame.draw.rect(screen, GROUND_BROWN, (x, y, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE), 1)

class Brick(Tile):
    def __init__(self, x, y):
        super().__init__(x, y, "brick")
        self.broken = False
        
    def hit(self, player, powerups, coins):
        if player.super and not self.broken:
            self.broken = True
            
    def draw(self, screen, camera):
        if not self.broken:
            x, y = camera.apply(self.x, self.y)
            pygame.draw.rect(screen, BRICK_RED, (x, y, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE), 1)
            pygame.draw.line(screen, BLACK, (x + 8, y), (x + 8, y + TILE_SIZE))
            pygame.draw.line(screen, BLACK, (x, y + 8), (x + TILE_SIZE, y + 8))

class QuestionBlock(Tile):
    def __init__(self, x, y, contents="coin"):
        super().__init__(x, y, "question")
        self.contents = contents
        self.used = False
        
    def hit(self, player, powerups, coins):
        if not self.used:
            self.used = True
            if self.contents == "coin":
                coins.append(Coin(self.x, self.y - TILE_SIZE * 3, True))
                player.coins += 1  # Increment coin counter
            elif self.contents == "mushroom":
                powerups.append(Powerup(self.x, self.y - TILE_SIZE, "mushroom"))
                
    def draw(self, screen, camera):
        x, y = camera.apply(self.x, self.y)
        color = (100, 100, 100) if self.used else QUESTION_YELLOW
        pygame.draw.rect(screen, color, (x, y, TILE_SIZE, TILE_SIZE))
        pygame.draw.rect(screen, BLACK, (x, y, TILE_SIZE, TILE_SIZE), 1)
        if not self.used:
            # Draw "?" symbol instead of dot
            font = pygame.font.Font(None, 20)
            text = font.render("?", True, BLACK)
            text_rect = text.get_rect(center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2))
            screen.blit(text, text_rect)

class Pipe:
    def __init__(self, x, y, height):
        self.x = x
        self.y = y
        self.height = height
        self.rects = [pygame.Rect(x, y + i * TILE_SIZE, TILE_SIZE * 2, TILE_SIZE) for i in range(height)]
        
    def draw(self, screen, camera):
        for i, rect in enumerate(self.rects):
            x, y = camera.apply(rect.x, rect.y)
            pygame.draw.rect(screen, PIPE_GREEN, (x, y, rect.width, rect.height))
            pygame.draw.rect(screen, BLACK, (x, y, rect.width, rect.height), 2)
            if i == 0:
                pygame.draw.rect(screen, PIPE_GREEN, (x - 2, y - 2, rect.width + 4, TILE_SIZE + 2))
                pygame.draw.rect(screen, BLACK, (x - 2, y - 2, rect.width + 4, TILE_SIZE + 2), 2)

class Coin:
    def __init__(self, x, y, flying=False):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x + 4, y + 4, 8, 8)
        self.flying = flying
        self.vy = -5 if flying else 0
        self.life = 30 if flying else -1
        self.collected = False
        
    def update(self):
        if self.flying:
            self.y += self.vy
            self.vy += 0.3
            self.rect.y = int(self.y) + 4
            self.life -= 1
            
    def collect(self):
        self.collected = True
        
    def draw(self, screen, camera):
        if self.life == 0 or self.collected:
            return
        x, y = camera.apply(self.rect.x, self.rect.y)
        pygame.draw.circle(screen, COIN_GOLD, (int(x + 4), int(y + 4)), 4)
        pygame.draw.circle(screen, BLACK, (int(x + 4), int(y + 4)), 4, 1)

class Powerup:
    def __init__(self, x, y, ptype):
        self.x = x
        self.y = y
        self.type = ptype
        self.vx = 2
        self.vy = 0
        self.width = TILE_SIZE
        self.height = TILE_SIZE
        
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
        
    def update(self, tiles):
        self.vy = min(self.vy + GRAVITY, MAX_FALL_SPEED)
        
        # Horizontal
        self.x += self.vx
        rect = self.get_rect()
        for tile in tiles:
            if rect.colliderect(tile.rect):
                if self.vx > 0:
                    self.x = tile.rect.left - self.width
                    self.vx = -2
                elif self.vx < 0:
                    self.x = tile.rect.right
                    self.vx = 2
                rect = self.get_rect()
                
        # Vertical
        self.y += self.vy
        rect = self.get_rect()
        for tile in tiles:
            if rect.colliderect(tile.rect):
                if self.vy > 0:
                    self.y = tile.rect.top - self.height
                    self.vy = 0
                elif self.vy < 0:
                    self.y = tile.rect.bottom
                    self.vy = 0
                rect = self.get_rect()
                
    def draw(self, screen, camera):
        x, y = camera.apply(self.x, self.y)
        if USE_SPRITES:
            # Use mushroom sprite - scale up 1.3x
            scaled_width = int(self.width * 1.3)
            scaled_height = int(self.height * 1.3)
            scaled_img = pygame.transform.scale(mushroom_img, (scaled_width, scaled_height))
            # Center horizontally, align to bottom
            offset_x = (scaled_width - self.width) // 2
            offset_y = scaled_height - self.height
            screen.blit(scaled_img, (x - offset_x, y - offset_y))
        else:
            # Fallback to rectangle
            pygame.draw.rect(screen, LUIGI_RED, (x, y, self.width, self.height))
            pygame.draw.circle(screen, WHITE, (int(x + 8), int(y + 8)), 3)

class Enemy:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = -1
        self.vy = 0
        self.width = TILE_SIZE
        self.height = TILE_SIZE
        
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)
        
    def update(self, tiles):
        self.vy = min(self.vy + GRAVITY, MAX_FALL_SPEED)
        
        # Check for edge/pit ahead before moving
        check_x = self.x + self.width if self.vx > 0 else self.x - TILE_SIZE
        check_rect = pygame.Rect(check_x, self.y + self.height, TILE_SIZE, TILE_SIZE)
        ground_ahead = False
        for tile in tiles:
            if check_rect.colliderect(tile.rect):
                ground_ahead = True
                break
        
        # Turn around at edges
        if not ground_ahead:
            self.vx = -self.vx
        
        # Horizontal
        self.x += self.vx
        rect = self.get_rect()
        for tile in tiles:
            if rect.colliderect(tile.rect):
                if self.vx > 0:
                    self.x = tile.rect.left - self.width
                    self.vx = -1
                elif self.vx < 0:
                    self.x = tile.rect.right
                    self.vx = 1
                rect = self.get_rect()
                
        # Vertical
        self.y += self.vy
        rect = self.get_rect()
        for tile in tiles:
            if rect.colliderect(tile.rect):
                if self.vy > 0:
                    self.y = tile.rect.top - self.height
                    self.vy = 0
                elif self.vy < 0:
                    self.y = tile.rect.bottom
                    self.vy = 0
                rect = self.get_rect()
                
    def stomped(self, enemies):
        if self in enemies:
            enemies.remove(self)

class Goomba(Enemy):
    def draw(self, screen, camera):
        x, y = camera.apply(self.x, self.y)
        if USE_SPRITES:
            # Use goomba sprite - scale up 1.3x
            scaled_width = int(self.width * 1.3)
            scaled_height = int(self.height * 1.3)
            scaled_img = pygame.transform.scale(goomba_img, (scaled_width, scaled_height))
            # Center horizontally, align to bottom
            offset_x = (scaled_width - self.width) // 2
            offset_y = scaled_height - self.height
            screen.blit(scaled_img, (x - offset_x, y - offset_y))
        else:
            # Fallback to rectangle
            pygame.draw.rect(screen, GOOMBA_BROWN, (x, y, self.width, self.height))
            pygame.draw.circle(screen, BLACK, (int(x + 5), int(y + 5)), 2)
            pygame.draw.circle(screen, BLACK, (int(x + 11), int(y + 5)), 2)

class Flagpole:
    def __init__(self, x):
        self.x = x
        self.y = SCREEN_HEIGHT - TILE_SIZE * 12  # Fixed: sits on ground now
        self.height = TILE_SIZE * 10
        self.touched = False
        
    def check_touch(self, player):
        pole_rect = pygame.Rect(self.x, self.y, TILE_SIZE, self.height)
        if pole_rect.colliderect(player.get_rect()) and not self.touched:
            self.touched = True
            return True
        return False
            
    def draw(self, screen, camera):
        x, y = camera.apply(self.x, self.y)
        pygame.draw.rect(screen, WHITE, (x, y, 2, self.height))
        # Static flag at top
        pygame.draw.rect(screen, FLAG_WHITE, (x + 2, y, TILE_SIZE, TILE_SIZE))

def create_level(rng=None):
    """Create level with optional randomness in enemy positions.
    
    Args:
        rng: numpy random generator for randomization (optional)
    """
    tiles = []
    enemies = []
    coins = []
    powerups = []
    
    # Ground segments
    for i in range(0, 69):
        tiles.append(Tile(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE, "ground"))
        tiles.append(Tile(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 2, "ground"))
        
    for i in range(72, 86):
        tiles.append(Tile(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE, "ground"))
        tiles.append(Tile(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 2, "ground"))
        
    for i in range(89, 154):
        tiles.append(Tile(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE, "ground"))
        tiles.append(Tile(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 2, "ground"))
        
    for i in range(158, 212):
        tiles.append(Tile(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE, "ground"))
        tiles.append(Tile(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 2, "ground"))
        
    # Starting bricks and ?-blocks
    for i in range(16, 21):
        tiles.append(QuestionBlock(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, "coin"))
    tiles.append(QuestionBlock(22 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, "mushroom"))
    
    for i in range(20, 24):
        tiles.append(Brick(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 10))
        
    # More bricks
    for i in range(77, 80):
        if i != 78:  # Skip brick at 78 - question block is here
            tiles.append(Brick(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6))
    tiles.append(QuestionBlock(78 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, "coin"))
    
    for i in range(80, 83):
        if i != 81:  # Skip brick at 81 - question block is here
            tiles.append(Brick(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 10))
    tiles.append(QuestionBlock(81 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 10, "coin"))
    
    for i in range(91, 94):
        tiles.append(QuestionBlock(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, "coin"))
        
    for i in range(100, 103):
        tiles.append(Brick(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6))
    tiles.append(QuestionBlock(106 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, "coin"))
    tiles.append(QuestionBlock(109 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, "coin"))
    tiles.append(QuestionBlock(109 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 10, "coin"))
    tiles.append(QuestionBlock(112 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, "coin"))
    
    for i in range(118, 125):
        if i != 121:  # Skip brick at 121 - mushroom question block is here
            tiles.append(Brick(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6))
    # Add second mushroom block here
    tiles.append(QuestionBlock(121 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, "mushroom"))
        
    for i in range(128, 131):
        if i != 129:  # Skip brick at 129 - question blocks are here
            tiles.append(Brick(i * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 10))
    tiles.append(QuestionBlock(129 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 10, "coin"))
    tiles.append(QuestionBlock(129 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, "coin"))
    
    # Pipes
    pipes = [
        Pipe(28 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4, 2),
        Pipe(38 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 5, 3),
        Pipe(46 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, 4),
        Pipe(57 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, 4),
        Pipe(163 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4, 2),
        Pipe(179 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 6, 4),
    ]
    
    # Create collision tiles for pipes (2 tiles wide per pipe segment)
    for pipe in pipes:
        for r in pipe.rects:
            tiles.append(Tile(r.x, r.y, "pipe"))  # Left tile
            tiles.append(Tile(r.x + TILE_SIZE, r.y, "pipe"))  # Right tile
        
    # End staircase
    for step in range(1, 9):
        for i in range(step):
            tiles.append(Tile((134 + step) * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * (3 + i), "ground"))
            
    for step in range(8, 0, -1):
        for i in range(step):
            tiles.append(Tile((143 + (8 - step)) * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * (3 + i), "ground"))
            
    # Enemies (with optional randomization)
    enemy_base_positions = [22, 40, 50, 82, 86, 97, 114, 130]
    
    for base_x in enemy_base_positions:
        # Add random offset of -1 to +1 tiles if RNG provided
        if rng is not None:
            offset = rng.integers(-TILE_SIZE, TILE_SIZE + 1)
        else:
            offset = 0
        enemies.append(Goomba((base_x * TILE_SIZE) + offset, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(133 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    
    # Enemies near end pipes (before flagpole)
    enemies.append(Goomba(165 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(175 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    
    flagpole = Flagpole(198 * TILE_SIZE)
    
    return tiles, enemies, coins, powerups, pipes, flagpole


class LuigiEnv(gym.Env):
    """
    Super Luigi Bros Gymnasium Environment
    
    Reward modes:
    - 'explorer': Rewards hitting question blocks and collecting coins
    - 'speedrunner': Rewards fast completion and forward progress
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None,
                 reward_mode: str = "explorer", reward_config_path: Optional[str] = None):
        super().__init__()
        self._rnd = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        
        # Load reward configuration
        if reward_config_path is None:
            # Default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            reward_config_path = os.path.join(current_dir, "..", "..", "configs", "rewards.yaml")
        
        try:
            with open(reward_config_path, 'r') as f:
                reward_configs = yaml.safe_load(f)
                self.reward_config = reward_configs[reward_mode]
        except Exception as e:
            print(f"Warning: Could not load reward config from {reward_config_path}: {e}")
            print("Using default hardcoded values.")
            # Fallback to hardcoded defaults
            self.reward_config = self._get_default_reward_config(reward_mode)
        
        # Actions: 0=noop, 1=left, 2=right, 3=jump, 4=right+jump, 5=sprint+right, 6=sprint+right+jump, 7=left+jump
        self.action_space = spaces.Discrete(8)
        
        # Observation: [player_x, player_y, player_vx, player_vy, player_super, on_ground, coins, 
        #               furthest_x, 9x7 tile grid (63), 5 nearest enemies (15), 3 nearest questions (9), flagpole_dx, completion]
        # Total: 8 + 63 + 15 + 9 + 2 = 97
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(97,), dtype=np.float32
        )
        
        self.max_steps = 10000
        
        # Episode metrics
        self._deaths = 0
        self._completions = 0
        self._coins_collected = 0
        self._blocks_hit = 0
        self._jumps = 0
        self._frames_alive = 0
        
        # Pygame lazy init
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        
    def _lazy_pygame(self):
        global USE_SPRITES, luigi_small_img, luigi_big_img, goomba_img, mushroom_img
        
        if self._pygame is None:
            pygame.init()
            self._pygame = True
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Super Luigi Bros - DRL")
            else:
                self._screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self._clock = pygame.time.Clock()
            self._font = pygame.font.Font(None, 16)
            
            # Load sprites after pygame is initialized
            if not USE_SPRITES:
                try:
                    luigi_small_img = pygame.image.load(os.path.join(SCRIPT_DIR, "luigi_small.png")).convert_alpha()
                    luigi_big_img = pygame.image.load(os.path.join(SCRIPT_DIR, "luigi_big.png")).convert_alpha()
                    goomba_img = pygame.image.load(os.path.join(SCRIPT_DIR, "goomba.png")).convert_alpha()
                    mushroom_img = pygame.image.load(os.path.join(SCRIPT_DIR, "mushroom.png")).convert_alpha()
                    USE_SPRITES = True
                except Exception as e:
                    USE_SPRITES = False
                    print(f"Warning: Could not load sprite images: {e}")
    
    def _get_default_reward_config(self, reward_mode: str) -> dict:
        """Fallback reward configuration if YAML fails to load"""
        if reward_mode == "explorer":
            return {
                'alive_bonus': 0.1,
                'new_ground_multiplier': 2.0,
                'new_ground_base': 0.1,
                'forward_multiplier': 0.08,
                'backward_penalty': -0.15,
                'standing_still_penalty': -0.1,
                'standing_still_threshold': 0.1,
                'jump_proximity_max_reward': 2.0,
                'jump_proximity_max_distance': 3,
                'question_block_reward': 15.0,
                'mushroom_block_reward': 25.0,
                'coin_reward': 5.0
            }
        else:  # speedrunner
            return {
                'new_ground_multiplier': 3.0,
                'new_ground_base': 0.1,
                'forward_multiplier': 0.25,
                'backward_penalty': -0.15,
                'standing_still_penalty': -0.15,
                'standing_still_threshold': 0.1,
                'time_penalty': -0.005
            }
            
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rnd.seed(seed)
            self._np_rng = np.random.default_rng(seed)
            
        # Create game state (pass RNG for enemy randomization)
        self.camera = Camera()
        self.player = Player(32, SCREEN_HEIGHT - TILE_SIZE * 5)
        self.tiles, self.enemies, self.coins_list, self.powerups, self.pipes, self.flagpole = create_level(self._np_rng)
        
        # Track question blocks
        self.total_question_blocks = sum(1 for t in self.tiles if isinstance(t, QuestionBlock))
        self.mushroom_block_positions = set()
        for tile in self.tiles:
            if isinstance(tile, QuestionBlock) and tile.contents == "mushroom":
                self.mushroom_block_positions.add((tile.x, tile.y))
        
        self.question_blocks_hit = set()
        self.mushroom_blocks_hit = set()
        
        # Reset metrics
        self.frames = 0
        self.furthest_x = self.player.x
        self._deaths = 0
        self._completions = 0
        self._coins_collected = 0
        self._blocks_hit = 0
        self._jumps = 0
        self._frames_alive = 0
        
        obs = self._get_obs()
        info = {
            "reward_mode": self.reward_mode,
            "total_question_blocks": self.total_question_blocks,
            "mushroom_blocks": len(self.mushroom_block_positions)
        }
        return obs, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Convert action to keys
        keys = self._action_to_keys(action)
        
        # Track jumps
        if keys[pygame.K_SPACE] and self.player.on_ground:
            self._jumps += 1
            
        # Store previous state
        prev_x = self.player.x
        prev_coins = self.player.coins
        prev_blocks_hit = len(self.question_blocks_hit)
        prev_mushroom_hit = len(self.mushroom_blocks_hit)
        prev_on_ground = self.player.on_ground
        
        # Update game using exact game code
        self.player.update(keys, self.tiles, self.enemies, self.coins_list, self.powerups, self.camera)
        
        for enemy in self.enemies:
            enemy.update(self.tiles)
            
        for powerup in self.powerups:
            powerup.update(self.tiles)
            
        for coin in self.coins_list[:]:
            coin.update()
            if coin.flying and coin.life == 0:
                self.coins_list.remove(coin)
                
        self.camera.update(self.player.x)
        
        # Track question blocks hit
        for tile in self.tiles:
            if isinstance(tile, QuestionBlock) and tile.used:
                pos = (tile.x, tile.y)
                if pos not in self.question_blocks_hit:
                    self.question_blocks_hit.add(pos)
                    self._blocks_hit += 1
                    if pos in self.mushroom_block_positions:
                        self.mushroom_blocks_hit.add(pos)
        
        # Track coins collected
        if self.player.coins > prev_coins:
            self._coins_collected += (self.player.coins - prev_coins)
        
        # Update furthest position
        self.furthest_x = max(self.furthest_x, self.player.x)
        
        # Calculate reward
        reward = self._calculate_step_reward(prev_x, prev_coins, prev_blocks_hit, prev_mushroom_hit, prev_on_ground)
        
        # Check termination
        terminated = False
        truncated = False
        
        if not self.player.alive:
            terminated = True
            self._deaths += 1
            reward -= 50
            
        if self.flagpole.check_touch(self.player):
            terminated = True
            self._completions += 1
            reward += self._calculate_completion_bonus()
            
        self.frames += 1
        self._frames_alive += 1
        
        if self.frames >= self.max_steps:
            truncated = True
            
        obs = self._get_obs()
        
        if self.render_mode == "human":
            self.render()
            
        info = {
            "reward_mode": self.reward_mode,
            "deaths": self._deaths,
            "completions": self._completions,
            "frames_alive": self._frames_alive,
            "coins_collected": self._coins_collected,
            "blocks_hit": self._blocks_hit,
            "mushroom_blocks_hit": len(self.mushroom_blocks_hit),
            "blocks_hit_ratio": self._blocks_hit / max(1, self.total_question_blocks),
            "max_x_reached": int(self.furthest_x),
            "jumps": self._jumps,
            "distance_traveled": int(self.player.x - 32)
        }
        
        return obs, float(reward), bool(terminated), bool(truncated), info
        
    def _action_to_keys(self, action: int) -> Dict[int, bool]:
        """Convert discrete action to pygame key dict"""
        keys = {
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
            pygame.K_SPACE: False,
            pygame.K_LSHIFT: False,
            pygame.K_RSHIFT: False
        }
        
        if action == 0:  # noop
            pass
        elif action == 1:  # left
            keys[pygame.K_LEFT] = True
        elif action == 2:  # right
            keys[pygame.K_RIGHT] = True
        elif action == 3:  # jump
            keys[pygame.K_SPACE] = True
        elif action == 4:  # right + jump
            keys[pygame.K_RIGHT] = True
            keys[pygame.K_SPACE] = True
        elif action == 5:  # sprint + right
            keys[pygame.K_LSHIFT] = True
            keys[pygame.K_RIGHT] = True
        elif action == 6:  # sprint + right + jump
            keys[pygame.K_LSHIFT] = True
            keys[pygame.K_RIGHT] = True
            keys[pygame.K_SPACE] = True
        elif action == 7:  # left + jump
            keys[pygame.K_LEFT] = True
            keys[pygame.K_SPACE] = True
            
        return keys
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        obs = []
        
        # Player state (8 values - added furthest_x)
        obs.extend([
            self.player.x / WORLD_WIDTH,
            self.player.y / SCREEN_HEIGHT,
            self.player.vx / 4.0,
            self.player.vy / 10.0,
            float(self.player.super),
            float(self.player.on_ground),
            self.player.coins / 100.0,
            self.furthest_x / WORLD_WIDTH  # Track max progress
        ])
        
        # 9x7 tile grid around player (63 values - wider to see more ahead)
        tile_grid = self._get_tile_grid()
        obs.extend(tile_grid)
        
        # Nearest 5 enemies (15 values - more enemy awareness)
        enemy_obs = self._get_nearest_enemies(5)
        obs.extend(enemy_obs)
        
        # Nearest 3 question blocks (9 values)
        question_obs = self._get_nearest_questions(3)
        obs.extend(question_obs)
        
        # Progress (2 values)
        flagpole_dx = (self.flagpole.x - self.player.x) / WORLD_WIDTH
        completion = float(self.flagpole.touched)
        obs.extend([flagpole_dx, completion])
        
        return np.array(obs, dtype=np.float32)
        
    def _get_tile_grid(self) -> list:
        """Get 9x7 grid of tiles around player (wider to see more ahead)"""
        grid = []
        center_x = int(self.player.x // TILE_SIZE)
        center_y = int(self.player.y // TILE_SIZE)
        
        for dy in range(-3, 4):  # 7 tiles vertically
            for dx in range(-2, 7):  # 9 tiles horizontally (more ahead, less behind)
                tile_x = center_x + dx
                tile_y = center_y + dy
                tile_type = self._get_tile_at(tile_x * TILE_SIZE, tile_y * TILE_SIZE)
                grid.append(float(tile_type) / 6.0)  # Normalize
                
        return grid
        
    def _get_tile_at(self, x: int, y: int) -> int:
        """Get tile type encoding at position"""
        TILE_EMPTY = 0
        TILE_GROUND = 1
        TILE_BRICK = 2
        TILE_QUESTION = 3
        TILE_QUESTION_USED = 4
        TILE_PIPE = 5
        
        for tile in self.tiles:
            if abs(tile.x - x) < TILE_SIZE/2 and abs(tile.y - y) < TILE_SIZE/2:
                if isinstance(tile, QuestionBlock):
                    return TILE_QUESTION_USED if tile.used else TILE_QUESTION
                elif isinstance(tile, Brick):
                    if tile.broken:
                        return TILE_EMPTY
                    return TILE_BRICK
                elif isinstance(tile, Tile):
                    return TILE_GROUND if tile.type == "ground" else TILE_PIPE
                    
        # Check pipes
        for pipe in self.pipes:
            if (pipe.x <= x < pipe.x + TILE_SIZE * 2 and
                pipe.y <= y < pipe.y + TILE_SIZE * pipe.height):
                return TILE_PIPE
                    
        return TILE_EMPTY
        
    def _get_nearest_enemies(self, n: int) -> list:
        """Get n nearest enemies"""
        obs = []
        enemy_dists = []
        for enemy in self.enemies:
            dx = enemy.x - self.player.x
            dy = enemy.y - self.player.y
            dist = dx*dx + dy*dy
            enemy_dists.append((dist, enemy))
            
        enemy_dists.sort(key=lambda x: x[0])
        
        for i in range(n):
            if i < len(enemy_dists):
                _, enemy = enemy_dists[i]
                obs.extend([
                    (enemy.x - self.player.x) / WORLD_WIDTH,
                    (enemy.y - self.player.y) / SCREEN_HEIGHT,
                    1.0  # alive
                ])
            else:
                obs.extend([0.0, 0.0, 0.0])
                
        return obs
        
    def _get_nearest_questions(self, n: int) -> list:
        """Get n nearest question blocks"""
        obs = []
        question_blocks = [t for t in self.tiles if isinstance(t, QuestionBlock)]
        
        block_dists = []
        for block in question_blocks:
            dx = block.x - self.player.x
            dy = block.y - self.player.y
            dist = dx*dx + dy*dy
            block_dists.append((dist, block))
            
        block_dists.sort(key=lambda x: x[0])
        
        for i in range(n):
            if i < len(block_dists):
                _, block = block_dists[i]
                obs.extend([
                    (block.x - self.player.x) / WORLD_WIDTH,
                    (block.y - self.player.y) / SCREEN_HEIGHT,
                    float(block.used)
                ])
            else:
                obs.extend([0.0, 0.0, 0.0])
                
        return obs
        
    def _calculate_step_reward(self, prev_x: float, prev_coins: int, 
                               prev_blocks_hit: int, prev_mushroom_hit: int, prev_on_ground: bool) -> float:
        """Calculate reward for this step"""
        reward = 0.0
        cfg = self.reward_config
        
        # Calculate movement
        dx = self.player.x - prev_x
        
        # Reward for reaching new furthest point
        new_ground_bonus = 0.0
        if self.player.x > self.furthest_x:
            new_ground_bonus = (self.player.x - self.furthest_x) * cfg['new_ground_base']
        
        if self.reward_mode == "explorer":
            # Alive bonus
            reward += cfg['alive_bonus']
            
            # STRONG reward for new ground
            reward += new_ground_bonus * cfg['new_ground_multiplier']
            
            # Reward for moving forward
            if dx > 0:
                reward += dx * cfg['forward_multiplier']
            elif dx < 0:
                reward += cfg['backward_penalty']
            
            # Detect if player just jumped (was on ground, now not)
            just_jumped = prev_on_ground and not self.player.on_ground and self.player.vy < 0
            
            # Reward jumping near question blocks
            if just_jumped:
                # Find nearby question blocks
                nearby_blocks = [t for t in self.tiles 
                               if isinstance(t, QuestionBlock) and not t.used
                               and abs(t.x - self.player.x) < TILE_SIZE * cfg['jump_proximity_max_distance']
                               and t.y < self.player.y]  # Blocks above player
                
                if nearby_blocks:
                    # Reward proportional to how close to blocks
                    closest_block = min(nearby_blocks, key=lambda b: abs(b.x - self.player.x))
                    distance = abs(closest_block.x - self.player.x)
                    proximity_reward = max(0, cfg['jump_proximity_max_reward'] - distance / TILE_SIZE)
                    reward += proximity_reward
            
            # Reward hitting question blocks
            blocks_hit_this_step = len(self.question_blocks_hit) - prev_blocks_hit
            if blocks_hit_this_step > 0:
                reward += cfg['question_block_reward'] * blocks_hit_this_step
                
            # Extra reward for mushroom blocks
            mushroom_hit_this_step = len(self.mushroom_blocks_hit) - prev_mushroom_hit
            if mushroom_hit_this_step > 0:
                reward += cfg['mushroom_block_reward'] * mushroom_hit_this_step
                
            # Reward coin collection
            coins_this_step = self.player.coins - prev_coins
            if coins_this_step > 0:
                reward += cfg['coin_reward'] * coins_this_step
            
            # Strong penalty for standing still (encourages exploration)
            if abs(dx) < cfg['standing_still_threshold']:
                reward += cfg['standing_still_penalty']
                
        else:  # speedrunner
            # STRONG reward for new ground
            reward += new_ground_bonus * cfg['new_ground_multiplier']
            
            # Strong reward for forward movement
            if dx > 0:
                reward += dx * cfg['forward_multiplier']
            else:
                reward += cfg['backward_penalty']
            
            # Strong penalty for standing still
            if abs(dx) < cfg['standing_still_threshold']:
                reward += cfg['standing_still_penalty']
            
            # Small time penalty (encourage speed)
            reward += cfg['time_penalty']
            
        return reward
        
    def _calculate_completion_bonus(self) -> float:
        """Calculate bonus for completing level"""
        if self.reward_mode == "explorer":
            # Huge bonus for hitting all blocks before completion
            blocks_ratio = len(self.question_blocks_hit) / max(1, self.total_question_blocks)
            if blocks_ratio >= 0.95:  # Hit 95%+ of blocks
                bonus = 200.0
            else:
                bonus = 50.0
            bonus += self.player.coins * 1.0
            return bonus
        else:  # speedrunner
            # Huge bonus for completion
            bonus = 200.0
            # Time bonus (faster = better)
            time_bonus = max(0, (10000 - self.frames) / 100.0)
            bonus += time_bonus
            return bonus
            
    def render(self):
        """Render the environment"""
        self._lazy_pygame()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        
        self._screen.fill(SKY_BLUE)
        
        # Draw tiles
        for tile in self.tiles:
            if not isinstance(tile, Brick) or not tile.broken:
                tile.draw(self._screen, self.camera)
                
        # Draw pipes
        for pipe in self.pipes:
            pipe.draw(self._screen, self.camera)
            
        # Draw coins
        for coin in self.coins_list:
            coin.draw(self._screen, self.camera)
            
        # Draw powerups
        for powerup in self.powerups:
            powerup.draw(self._screen, self.camera)
            
        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self._screen, self.camera)
            
        # Draw player
        self.player.draw(self._screen, self.camera)
        
        # Draw flagpole
        self.flagpole.draw(self._screen, self.camera)
        
        # Draw HUD
        coin_text = self._font.render(f"COINS: x{self.player.coins:02d}", True, WHITE)
        self._screen.blit(coin_text, (10, 10))
        
        blocks_text = self._font.render(
            f"BLOCKS: {len(self.question_blocks_hit)}/{self.total_question_blocks}",
            True, WHITE
        )
        self._screen.blit(blocks_text, (10, 25))
        
        mode_text = self._font.render(f"MODE: {self.reward_mode.upper()}", True, WHITE)
        self._screen.blit(mode_text, (SCREEN_WIDTH - 120, 10))
        
        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(self._screen)
            return np.transpose(arr, (1, 0, 2))
            
    def close(self):
        """Clean up resources"""
        if self._pygame is not None:
            pygame.quit()
            self._pygame = None
            self._screen = None
            self._clock = None
            self._font = None


def main():
    """Standalone playable version"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Super Luigi Bros")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 16)
    
    camera = Camera()
    player = Player(32, SCREEN_HEIGHT - TILE_SIZE * 5)
    tiles, enemies, coins, powerups, pipes, flagpole = create_level()
    
    time_left = 400
    frame_count = 0
    elapsed_frames = 0
    game_state = "playing"
    
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # Restart level when R is pressed and course is cleared
                if event.key == pygame.K_r and game_state == "course_clear":
                    player = Player(32, SCREEN_HEIGHT - TILE_SIZE * 5)
                    tiles, enemies, coins, powerups, pipes, flagpole = create_level()
                    time_left = 400
                    frame_count = 0
                    elapsed_frames = 0
                    camera.x = 0
                    camera.max_x = 0
                    game_state = "playing"
                    
        keys = pygame.key.get_pressed()
        
        if game_state == "playing":
            # Update player
            player.update(keys, tiles, enemies, coins, powerups, camera)
            
            # Update enemies
            for enemy in enemies:
                enemy.update(tiles)
                                
            # Update powerups
            for powerup in powerups:
                powerup.update(tiles)
                
            # Update coins
            for coin in coins[:]:
                coin.update()
                if coin.flying and coin.life == 0:
                    coins.remove(coin)
                    
            # Update camera
            camera.update(player.x)
            
            # Update time
            frame_count += 1
            elapsed_frames += 1  # Increment total elapsed time
            if frame_count >= FPS:
                frame_count = 0
                time_left -= 1
                if time_left <= 0:
                    player.alive = False
                    
            # Check player death - auto-restart immediately
            if not player.alive:
                # Respawn
                player = Player(32, SCREEN_HEIGHT - TILE_SIZE * 5)
                tiles, enemies, coins, powerups, pipes, flagpole = create_level()
                time_left = 400
                frame_count = 0
                elapsed_frames = 0  # Reset timer on death
                camera.x = 0
                camera.max_x = 0
                    
            # Check flagpole - immediately trigger course clear
            if flagpole.check_touch(player):
                game_state = "course_clear"
                
        # Draw
        screen.fill(SKY_BLUE)
        
        # Draw tiles
        for tile in tiles:
            if not isinstance(tile, Brick) or not tile.broken:
                tile.draw(screen, camera)
                
        # Draw pipes
        for pipe in pipes:
            pipe.draw(screen, camera)
            
        # Draw coins
        for coin in coins:
            coin.draw(screen, camera)
            
        # Draw powerups
        for powerup in powerups:
            powerup.draw(screen, camera)
            
        # Draw enemies
        for enemy in enemies:
            enemy.draw(screen, camera)
            
        # Draw player
        player.draw(screen, camera)
        
        # Draw flagpole
        flagpole.draw(screen, camera)
        
        # Draw coin counter HUD at top
        coin_text = font.render(f"COINS: x{player.coins:02d}", True, WHITE)
        screen.blit(coin_text, (10, 10))
        
        # Draw timer HUD at top right
        elapsed_seconds = elapsed_frames / FPS
        timer_text = font.render(f"TIME: {elapsed_seconds:.1f}s", True, WHITE)
        screen.blit(timer_text, (SCREEN_WIDTH - 110, 10))
        
        # Draw game state messages
        if game_state == "course_clear":
            text = font.render("COURSE CLEAR!", True, WHITE)
            screen.blit(text, (SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2))
            text2 = font.render("R to restart | ESC to quit", True, WHITE)
            screen.blit(text2, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 + 20))
            
        pygame.display.flip()
        
    pygame.quit()

if __name__ == "__main__":
    main()
