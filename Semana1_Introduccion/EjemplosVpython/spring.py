from vpython import *

canvas(background=color.white)

ball = sphere(pos=vector(0.01, 0, 0), radius=0.002, color=color.red)
wall = box(pos=vector(-0.02, 0, 0), size=vector(0.001, 0.006, 0.006))
spring = helix(pos=wall.pos, axis=ball.pos-wall.pos, thickness=0.0002, radius=0.001)

m = 0.1        # Masa de la esfera
k = 10         # Constante del resorte
dt = 0.01      # Paso de tiempo
x = 0.01       # Posición inicial
v = 0          # Velocidad inicial
t = 0          # Tiempo inicial

while t < 3:
    rate(100)
    F = -k * x
    v = v + (F / m) * dt
    x = x + v * dt
    ball.pos = vector(x, 0, 0)
    spring.axis = ball.pos - wall.pos
    t = t + dt

