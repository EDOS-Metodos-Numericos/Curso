from vpython import *

canvas(background=color.white)

m = 0.03       # masa de la esfera
L0 = 0.05      # longitud natural del resorte
k = 10         # constante del resorte
g = vector(0, -9.8, 0)  # gravedad

top = box(pos=vector(0, L0/2, 0), size=vector(L0/2, L0/10, L0/2))
ball = sphere(pos=top.pos + vector(0, -L0, 0), radius=0.005, color=color.red, make_trail=True)

spring = helix(pos=top.pos, axis=ball.pos - top.pos, radius=0.002,
               coils=10, thickness=0.001)

ball.p = m * vector(0, 0, 0)  # momento lineal inicial
t = 0
dt = 0.01

while t < 4:
    rate(100)
    L = ball.pos - top.pos
    Fnet = m * g - k * (mag(L) - L0) * norm(L)
    ball.p = ball.p + Fnet * dt
    ball.pos = ball.pos + ball.p * dt / m
    t = t + dt
    spring.axis = ball.pos - top.pos
