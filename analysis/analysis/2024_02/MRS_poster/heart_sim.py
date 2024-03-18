import pygame
import numpy as np

# Simulation parameters
width, height = 400, 200
scale = 2
num_spheres = 5
viscosity = 0.02
density = 1.0
tau = 3 * viscosity + 0.5

# Lattice weights and velocities for D2Q9
w = np.array([4/9] + [1/9] * 4 + [1/36] * 4)
dx = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]])

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((width * scale, height * scale))
clock = pygame.time.Clock()

# Initialize populations
populations = np.ones((width, height, 9)) * density

# Initialize spheres
spheres = [{'x': np.random.randint(20, width - 20),
            'y': np.random.randint(20, height - 20),
            'vx': np.random.uniform(-0.1, 0.1),
            'vy': np.random.uniform(-0.1, 0.1),
            'radius': np.random.randint(5, 15)} for _ in range(num_spheres)]

def equilibrium_distribution(rho, u):
    cu = 3 * (dx[0, 1:] * u[:, :, 0, None] + dx[1, 1:] * u[:, :, 1, None])
    feq = rho * (w[None, None, :] * (1 + cu + 0.5 * cu**2 - 3/2 * (u[:, :, 0]**2 + u[:, :, 1]**2))).T
    return feq

def collide():
    global populations

    rho = np.sum(populations[:, :, 1:], axis=2) + populations[:, :, 0]
    u = np.zeros((width, height, 2))
    for i in range(2):
        u[:, :, i] = np.sum(dx[i, 1:] * populations[:, :, 1:], axis=2) / rho

    feq = equilibrium_distribution(rho, u)

    for i in range(9):
        populations[:, :, i] += 1/tau * (feq[i] - populations[:, :, i])

def propagate():
    global populations

    for i in range(1, 9):
        populations[:, :, i] = np.roll(populations[:, :, i], dx[:, i], axis=(0, 1))

def apply_boundary_conditions():
    global populations

    # Example: Add obstacles (spheres)
    for sphere in spheres:
        x, y, radius = sphere['x'], sphere['y'], sphere['radius']
        for i in range(width):
            for j in range(height):
                if (i - x)**2 + (j - y)**2 < radius**2:
                    populations[i, j, :] = 0

# Main simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Collision and propagation steps
    collide()
    propagate()

    # Move spheres
    for sphere in spheres:
        sphere['x'] += sphere['vx']
        sphere['y'] += sphere['vy']

    # Apply boundary conditions
    apply_boundary_conditions()

    # Visualization
    screen.fill((255, 255, 255))
    for x in range(width):
        for y in range(height):
            total_color = int(np.clip(np.sum(populations[x, y, :]), 0, 255))
            color = (total_color, total_color, total_color)
            pygame.draw.rect(screen, color, (x * scale, y * scale, scale, scale))

    # Draw spheres
    for sphere in spheres:
        pygame.draw.circle(screen, (0, 0, 255), (int(sphere['x'] * scale), int(sphere['y'] * scale)), int(sphere['radius'] * scale))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
