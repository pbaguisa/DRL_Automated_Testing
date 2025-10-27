import pygame
import os

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

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Super Luigi Bros")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 16)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load sprites (with fallback to rectangles if images don't exist)
try:
    luigi_small_img = pygame.image.load(os.path.join(SCRIPT_DIR, "luigi_small.png")).convert_alpha()
    luigi_big_img = pygame.image.load(os.path.join(SCRIPT_DIR, "luigi_big.png")).convert_alpha()
    goomba_img = pygame.image.load(os.path.join(SCRIPT_DIR, "goomba.png")).convert_alpha()
    mushroom_img = pygame.image.load(os.path.join(SCRIPT_DIR, "mushroom.png")).convert_alpha()
    USE_SPRITES = True
except:
    USE_SPRITES = False
    print("Warning: Could not load sprite images. Using placeholder rectangles.")

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
                if self.vy > 0 and rect.bottom <= enemy_rect.centery:
                    # Stomp
                    enemy.stomped(enemies)
                    self.vy = JUMP_IMPULSE * 0.5
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

def create_level():
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
            
    # Enemies
    enemies.append(Goomba(22 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(40 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(50 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(82 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(86 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(97 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(114 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(130 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(133 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    
    # Enemies near end pipes (before flagpole)
    enemies.append(Goomba(165 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    enemies.append(Goomba(175 * TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE * 4))
    
    flagpole = Flagpole(198 * TILE_SIZE)
    
    return tiles, enemies, coins, powerups, pipes, flagpole

def main():
    camera = Camera()
    player = Player(32, SCREEN_HEIGHT - TILE_SIZE * 5)
    tiles, enemies, coins, powerups, pipes, flagpole = create_level()
    
    time_left = 400
    frame_count = 0
    elapsed_frames = 0  # Track total elapsed frames for timer
    game_state = "playing"  # playing, course_clear
    
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
