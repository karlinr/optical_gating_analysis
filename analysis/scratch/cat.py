import turtle
import colorsys

turtle.speed(0)  # Fastest drawing speed
turtle.bgcolor("black")
turtle.hideturtle()

colors = ["red", "orange", "yellow", "green", "blue", "purple"]
num_colors = len(colors)
angle = 17  # Angle for slight asymmetry, making it more trippy

def draw_spiral(size, color_index):
    turtle.color(colors[color_index])
    for _ in range(100):
        turtle.forward(size)
        turtle.left(angle)
        size += 0.5   

size = 1
color_index = 0

while True:
    draw_spiral(size, color_index)
    size += 5
    color_index = (color_index + 1) % num_colors  # Cycle through colors

turtle.done()
