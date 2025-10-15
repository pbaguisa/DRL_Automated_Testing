import pygame
import math

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bubble Trouble - Fixed Vertical Harpoon")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

clock = pygame.time.Clock()
FPS = 60

# Player class
class Player:
    def __init__(self):
        self.width = 30
        self.height = 50
        self.x = WIDTH // 2
        self.y = HEIGHT - self.height - 10
        self.speed = 4
        self.color = BLUE

    def move(self, keys):
        if keys[pygame.K_LEFT]:
            self.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.x += self.speed
        self.x = max(0, min(self.x, WIDTH - self.width))

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))

    def get_center(self):
        return (self.x + self.width // 2, self.y)

    def get_edges(self):
        return (self.x, self.x + self.width, self.y, self.y + self.height)

# Bullet (Harpoon tip) class
class Bullet:
    def __init__(self, x, y):
        self.x = x  # fixed x position where shot was fired
        self.y = y
        self.radius = 6
        self.speed = 10
        self.color = RED
        self.active = True

    def update(self):
        self.y -= self.speed
        if self.y < 0:
            self.active = False

    def draw(self):
        # Draw bullet tip
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
    
    def line_start(self):
        # Line starts at bottom of screen at fixed x (where bullet was fired)
        return (self.x, HEIGHT)
    
    def line_end(self):
        # Line ends at bullet tip (x, y)
        return (self.x, self.y)

# Bubble class
class Bubble:
    def __init__(self, x, y, size, x_vel, y_vel):
        self.x = x
        self.y = y
        self.size = size  # radius
        self.color = GREEN
        self.x_vel = x_vel * 0.5  # Slower horizontal speed
        self.y_vel = y_vel * 0.5  # Slower vertical speed
        self.gravity = 0.3

    def update(self):
        # Move bubble
        self.x += self.x_vel
        self.y += self.y_vel

        # Bounce off walls
        if self.x - self.size < 0 or self.x + self.size > WIDTH:
            self.x_vel *= -1

        # Bounce off floor and ceiling with constant vertical momentum
        if self.y + self.size > HEIGHT:
            self.y = HEIGHT - self.size
            self.y_vel = -abs(self.y_vel)  # reverse velocity without damping
        else:
            self.y_vel += self.gravity  # gravity pulling down

        if self.y - self.size < 0:
            self.y = self.size
            self.y_vel = abs(self.y_vel)  # bounce down

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

    def collide_with_point(self, px, py):
        dist = math.hypot(self.x - px, self.y - py)
        return dist < self.size

    def collide_with_line(self, x1, y1, x2, y2):
        # Check distance from bubble center to line segment (x1,y1)-(x2,y2)
        px = self.x
        py = self.y
        line_dx = x2 - x1
        line_dy = y2 - y1
        if line_dx == 0 and line_dy == 0:
            # line is a point
            return self.collide_with_point(x1, y1)

        t = ((px - x1) * line_dx + (py - y1) * line_dy) / (line_dx**2 + line_dy**2)
        t = max(0, min(1, t))  # clamp to segment

        closest_x = x1 + t * line_dx
        closest_y = y1 + t * line_dy

        dist = math.hypot(px - closest_x, py - closest_y)
        return dist < self.size

    def collide_with_player(self, player):
        player_left, player_right, player_top, player_bottom = player.get_edges()

        # Check for overlap between bubble and player’s rectangle
        # Check distance from player’s rectangle edges to the bubble’s perimeter
        dist_x = abs(self.x - (player_left + player_right) / 2)
        dist_y = abs(self.y - (player_top + player_bottom) / 2)

        if dist_x > (player_right - player_left) / 2 + self.size:
            return False
        if dist_y > (player_bottom - player_top) / 2 + self.size:
            return False

        if dist_x <= (player_right - player_left) / 2 or dist_y <= (player_bottom - player_top) / 2:
            return True

        corner_distance_sq = (dist_x - (player_right - player_left) / 2) ** 2 + \
                             (dist_y - (player_bottom - player_top) / 2) ** 2
        return corner_distance_sq <= self.size ** 2

# Game variables
player = Player()
bullet = None  # only one bullet allowed at a time
bubbles = []

# Function to reset the game
def reset_game():
    global player, bullet, bubbles, level
    player = Player()  # Reset the player position and stats
    bullet = None  # No bullet at the start
    bubbles = [Bubble(100, 100, 40, 2, -3), Bubble(500, 150, 40, -1, -2)]  # Reset bubbles
    level = 1  # Reset level

# Initial setup
reset_game()

# Main game loop
running = True
while running:
    clock.tick(FPS)
    screen.fill(BLACK)

    keys = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and bullet is None:
                # Shoot bullet from player's current center position (fixed x)
                bullet_x, bullet_y = player.get_center()
                bullet = Bullet(bullet_x, bullet_y)

    # Update bullet + rope line (draw behind player)
    if bullet:
        bullet.update()
        if not bullet.active:
            bullet = None
        else:
            # Draw rope line from bottom of screen up to bullet tip (fixed x)
            line_start = bullet.line_start()
            line_end = bullet.line_end()
            pygame.draw.line(screen, WHITE, line_start, line_end, 3)
            bullet.draw()

    # Update and draw player (draw on top)
    player.move(keys)
    player.draw()

    # Update and draw bubbles
    for bubble in bubbles[:]:
        bubble.update()
        bubble.draw()

        # Check for collision with the player (end the game if hit)
        if bubble.collide_with_player(player):
            print("Game Over: Player hit by bubble! Restarting the game.")
            reset_game()  # Restart the game
            break

        # Check collision with bullet tip and rope if bullet active
        if bullet:
            # Check bullet tip collision
            if bubble.collide_with_point(bullet.x, bullet.y):
                # Pop or split bubble
                if bubble.size > 15:
                    new_size = bubble.size // 2
                    bubbles.append(Bubble(bubble.x, bubble.y, new_size, -abs(bubble.x_vel), -8))
                    bubbles.append(Bubble(bubble.x, bubble.y, new_size, abs(bubble.x_vel), -8))
                bubbles.remove(bubble)
                bullet = None
                break
            else:
                # Check rope collision
                line_start = bullet.line_start()
                line_end = bullet.line_end()
                if bubble.collide_with_line(*line_start, *line_end):
                    if bubble.size > 15:
                        new_size = bubble.size // 2
                        bubbles.append(Bubble(bubble.x, bubble.y, new_size, -abs(bubble.x_vel), -8))
                        bubbles.append(Bubble(bubble.x, bubble.y, new_size, abs(bubble.x_vel), -8))
                    bubbles.remove(bubble)
                    bullet = None
                    break

    # Check if all bubbles are popped and move to the next level
    if len(bubbles) == 0:
        print(f"Level {level} completed!")
        level += 1
        # Reset the game for the next level
        bubbles.append(Bubble(100, 100, 40, 2, -3))  # Add new bubbles
        bubbles.append(Bubble(500, 150, 40, -1, -2))

    pygame.display.flip()

pygame.quit()
