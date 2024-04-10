import turtle
import colorsys
import numpy as np

turtle.speed(0)  # Fastest drawing speed
turtle.bgcolor("black")
turtle.hideturtle()

colors = ["red", "orange", "yellow", "green", "blue", "purple"]
num_colors = len(colors)
angle = 29  # Angle for slight asymmetry, making it more trippy

def draw_spiral(size, color_index):
    turtle.color(colors[color_index])
    for _ in range(100):
        turtle.forward(size)
        turtle.left(angle)
        size +=1

size = 1
color_index = 0

while True:
    draw_spiral(size, color_index)
    size += 5
    color_index = (color_index + 1) % num_colors  # Cycle through colors
    angle = np.sin(size / 10) * 25 + 25  # Sine wave for angle

turtle.done()
