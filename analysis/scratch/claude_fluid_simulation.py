import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Set up display
width, height = 1024, 768
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Fluid Simulation with Object and Force")

# Simulation parameters
N = 256  # grid size
dt = 0.1
diffusion = 0.05
viscosity = 0.05

# Create fluid arrays
velocity = np.zeros((N, N, 2), dtype=np.float32)
density = np.zeros((N, N), dtype=np.float32)

# Create object mask (circular object)
object_radius = N // 10
object_center = (N // 2, N // 2)
y, x = np.ogrid[:N, :N]
object_mask = ((x - object_center[0])**2 + (y - object_center[1])**2 <= object_radius**2)

# Constant force (e.g., gravity)
force = np.array([0.5, 0.5])  # Diagonal force (down and right)

# Create a persistent surface for fluid rendering
fluid_surface = pygame.Surface((N, N))

def add_density(x, y, amount):
    density[int(y * N), int(x * N)] += amount

def add_velocity(x, y, amount_x, amount_y):
    velocity[int(y * N), int(x * N)] += [amount_x, amount_y]

def diffuse(field, diff):
    field += diff * (np.roll(field, 1, 0) + np.roll(field, -1, 0) + 
                     np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)
    return field

def advect(field, velocity):
    h = 1.0 / N
    y, x = np.meshgrid(np.arange(N), np.arange(N))
    x0 = np.clip(x - velocity[:,:,0] * h * N, 0, N-1.01)
    y0 = np.clip(y - velocity[:,:,1] * h * N, 0, N-1.01)
    
    i0 = np.floor(x0).astype(int)
    i1 = i0 + 1
    j0 = np.floor(y0).astype(int)
    j1 = j0 + 1
    
    s1 = x0 - i0
    s0 = 1 - s1
    t1 = y0 - j0
    t0 = 1 - t1
    
    return (s0 * (t0 * field[j0, i0] + t1 * field[j1, i0]) +
            s1 * (t0 * field[j0, i1] + t1 * field[j1, i1]))

def apply_force(velocity, force):
    velocity[:,:,0] += force[0] * dt
    velocity[:,:,1] += force[1] * dt
    return velocity

def handle_object_collision(velocity, density):
    velocity[object_mask] = 0
    boundary = np.zeros_like(object_mask)
    boundary[1:-1, 1:-1] = object_mask[1:-1, 1:-1] ^ object_mask[2:, 1:-1] | \
                           object_mask[1:-1, 1:-1] ^ object_mask[:-2, 1:-1] | \
                           object_mask[1:-1, 1:-1] ^ object_mask[1:-1, 2:] | \
                           object_mask[1:-1, 1:-1] ^ object_mask[1:-1, :-2]
    velocity[boundary] *= -0.5
    density[object_mask] = 0
    return velocity, density

def fluid_step():
    global velocity, density
    velocity = apply_force(velocity, force)
    velocity[:,:,0] = diffuse(velocity[:,:,0], viscosity)
    velocity[:,:,1] = diffuse(velocity[:,:,1], viscosity)
    velocity[:,:,0] = advect(velocity[:,:,0], velocity)
    velocity[:,:,1] = advect(velocity[:,:,1], velocity)
    density = diffuse(density, diffusion)
    density = advect(density, velocity)
    velocity, density = handle_object_collision(velocity, density)
    density *= 0.99
    velocity *= 0.99

def update_fluid_surface():
    global fluid_surface
    
    # Calculate color based on velocity magnitude and direction
    velocity_magnitude = np.sqrt(np.sum(velocity**2, axis=2))
    max_velocity = np.max(velocity_magnitude)
    if max_velocity > 0:
        normalized_velocity = velocity_magnitude / max_velocity
    else:
        normalized_velocity = velocity_magnitude
    
    # Create a color wheel effect
    angle = np.arctan2(velocity[:,:,1], velocity[:,:,0])
    hue = (angle + np.pi) / (2 * np.pi)
    saturation = normalized_velocity
    value = np.clip(density * 3, 0, 1)
    
    # Convert HSV to RGB
    c = saturation * value
    x = c * (1 - abs((hue * 6) % 2 - 1))
    m = value - c
    
    rgb = np.zeros((N, N, 3))
    mask = (hue < 1/6)
    rgb[mask] = [c[mask], x[mask], 0]
    mask = (1/6 <= hue) & (hue < 2/6)
    rgb[mask] = [x[mask], c[mask], 0]
    mask = (2/6 <= hue) & (hue < 3/6)
    rgb[mask] = [0, c[mask], x[mask]]
    mask = (3/6 <= hue) & (hue < 4/6)
    rgb[mask] = [0, x[mask], c[mask]]
    mask = (4/6 <= hue) & (hue < 5/6)
    rgb[mask] = [x[mask], 0, c[mask]]
    mask = (5/6 <= hue)
    rgb[mask] = [c[mask], 0, x[mask]]
    rgb += m[:,:,np.newaxis]
    
    rgb_array = (rgb * 255).astype(np.uint8)
    
    # Add object to the visualization
    rgb_array[object_mask] = [255, 255, 255]  # White color for the object
    
    pygame.surfarray.blit_array(fluid_surface, rgb_array)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            dx, dy = event.rel
            add_density(x / width, y / height, 1)
            add_velocity(x / width, y / height, dx * 0.5, dy * 0.5)

    fluid_step()
    update_fluid_surface()

    # Draw fluid
    scaled_surface = pygame.transform.scale(fluid_surface, (width, height))
    screen.blit(scaled_surface, (0, 0))

    pygame.display.flip()
    clock.tick(60)
    print(f"FPS: {clock.get_fps():.2f}")

pygame.quit()