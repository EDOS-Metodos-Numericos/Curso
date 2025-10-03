import math
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import (
    Play, FloatSlider, FloatText, Checkbox, HBox, VBox, jslink, Layout
)
from IPython.display import display

plt.ioff()

# ------------------------ Dinámica y RK4 -------------------------------------
def dinamica_doble_pendulo(th1, th2, w1, w2, L1, L2, m1, m2, g):
    """
    Ecuaciones clásicas del péndulo doble (ángulos respecto a la vertical).
    Devuelve (dth1, dth2, dw1, dw2).
    """
    s12 = math.sin(th1 - th2)
    c12 = math.cos(th1 - th2)
    den = 2*m1 + m2 - m2*math.cos(2*th1 - 2*th2)

    dth1 = w1
    dth2 = w2

    dw1 = (
        -g*(2*m1 + m2)*math.sin(th1)
        - m2*g*math.sin(th1 - 2*th2)
        - 2*m2*s12*(w2*w2*L2 + w1*w1*L1*c12)
    ) / (L1 * den)

    dw2 = (
        2*s12*( w1*w1*L1*(m1+m2) + g*(m1+m2)*math.cos(th1) + w2*w2*L2*m2*c12 )
    ) / (L2 * den)

    return dth1, dth2, dw1, dw2


def integrar_doble_pendulo_rk4(theta1_0, theta2_0, w1_0, w2_0,
                               L1, L2, m1, m2, g, dt, n_steps):
    """
    Integra el sistema con RK4 y regresa arrays (th1, th2, w1, w2) de longitud n_steps+1.
    """
    th1 = np.empty(n_steps + 1, dtype=float)
    th2 = np.empty(n_steps + 1, dtype=float)
    w1  = np.empty(n_steps + 1, dtype=float)
    w2  = np.empty(n_steps + 1, dtype=float)

    th1[0], th2[0], w1[0], w2[0] = theta1_0, theta2_0, w1_0, w2_0

    for n in range(n_steps):
        y1 = (th1[n], th2[n], w1[n], w2[n])

        k1 = dinamica_doble_pendulo(*y1, L1, L2, m1, m2, g)

        y2 = (y1[0] + 0.5*dt*k1[0],
              y1[1] + 0.5*dt*k1[1],
              y1[2] + 0.5*dt*k1[2],
              y1[3] + 0.5*dt*k1[3])
        k2 = dinamica_doble_pendulo(*y2, L1, L2, m1, m2, g)

        y3 = (y1[0] + 0.5*dt*k2[0],
              y1[1] + 0.5*dt*k2[1],
              y1[2] + 0.5*dt*k2[2],
              y1[3] + 0.5*dt*k2[3])
        k3 = dinamica_doble_pendulo(*y3, L1, L2, m1, m2, g)

        y4 = (y1[0] + dt*k3[0],
              y1[1] + dt*k3[1],
              y1[2] + dt*k3[2],
              y1[3] + dt*k3[3])
        k4 = dinamica_doble_pendulo(*y4, L1, L2, m1, m2, g)

        th1[n+1] = th1[n] + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        th2[n+1] = th2[n] + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        w1[n+1]  = w1[n]  + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        w2[n+1]  = w2[n]  + (dt/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

    return th1, th2, w1, w2


def dobles_pendulos_interactivo():
    # ---------- Parámetros físicos compartidos ----
    slider_kw = dict(step=0.001, readout_format='.3f', continuous_update=True)

    L1 = FloatSlider(value=1.0, min=0.3, max=3.0, description='l₁', **slider_kw)
    L2 = FloatSlider(value=1.0, min=0.3, max=3.0, description='l₂', **slider_kw)
    m1 = FloatSlider(value=1.0, min=0.1, max=5.0, description='m₁', **slider_kw)
    m2 = FloatSlider(value=1.0, min=0.1, max=5.0, description='m₂', **slider_kw)
    g  = FloatSlider(value=9.81, min=1.0, max=20.0, description='g',  **slider_kw)

    # ϵ para péndulo B 
    eps_L1 = FloatText(value=0.0, description='ϵ_l₁', layout=Layout(width='170px'))
    eps_L2 = FloatText(value=0.0, description='ϵ_l₂', layout=Layout(width='170px'))
    eps_m1 = FloatText(value=0.0, description='ϵ_m₁', layout=Layout(width='170px'))
    eps_m2 = FloatText(value=0.0, description='ϵ_m₂', layout=Layout(width='170px'))
    eps_g  = FloatText(value=0.0, description='ϵ_g',  layout=Layout(width='170px'))

    # Tiempo
    dt = 0.01
    play = Play(min=0, max=3200, step=1, interval=10)
    t = FloatSlider(value=0.0, min=0.0, max=float(play.max), step=0.01,
                    description='t (s/100)', readout_format='.3f', continuous_update=True)
    jslink((play, 'value'), (t, 'value'))

    # Condiciones iniciales A 
    th1A = FloatSlider(value=+0.9, min=-math.pi, max=math.pi, description='θ₁A', **slider_kw)
    th2A = FloatSlider(value=-0.9, min=-math.pi, max=math.pi, description='θ₂A', **slider_kw)
    w1A  = FloatSlider(value=0.0,  min=-20.0, max=20.0,  description='ω₁A',  **slider_kw)
    w2A  = FloatSlider(value=0.0,  min=-20.0, max=20.0,  description='ω₂A',  **slider_kw)

    # Condiciones iniciales B + epsilon
    mostrar_B = Checkbox(value=True, description='Mostrar péndulo B')
    th1B = FloatSlider(value=th1A.value + 1e-3, min=-math.pi, max=math.pi, description='θ₁B', **slider_kw)
    th2B = FloatSlider(value=th2A.value,       min=-math.pi, max=math.pi, description='θ₂B', **slider_kw)
    w1B  = FloatSlider(value=0.0,  min=-20.0, max=20.0, description='ω₁B', **slider_kw)
    w2B  = FloatSlider(value=0.0,  min=-20.0, max=20.0, description='ω₂B', **slider_kw)

    eps_th1B = FloatText(value=0.0, description='ϵ_θ₁B', layout=Layout(width='170px'))
    eps_th2B = FloatText(value=0.0, description='ϵ_θ₂B', layout=Layout(width='170px'))
    eps_w1B  = FloatText(value=0.0, description='ϵ_ω₁B',  layout=Layout(width='170px'))
    eps_w2B  = FloatText(value=0.0, description='ϵ_ω₂B',  layout=Layout(width='170px'))

    # ---------------------- Figura ---------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.responsive = True
    except Exception:
        pass
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    def Ltot_max():
        LtotA = L1.value + L2.value
        LtotB = (L1.value*(1+eps_L1.value)) + (L2.value*(1+eps_L2.value))
        return 1.2*max(LtotA, LtotB)

    def fijar_limites():
        Ltot = Ltot_max()
        ax.set_xlim(-Ltot, Ltot); ax.set_ylim(-Ltot, Ltot)

    fijar_limites()

    # Péndulo A
    lineA1, = ax.plot([0, 0], [0, -L1.value], lw=2)
    lineA2, = ax.plot([0, 0], [-L1.value, -(L1.value+L2.value)], lw=2)
    bobA1 = plt.Circle((0, -L1.value), radius=0.05*L1.value*m1.value, alpha=0.85)
    bobA2 = plt.Circle((0, -(L1.value+L2.value)), radius=0.05*L2.value*m2.value, alpha=0.85)
    ax.add_patch(bobA1); ax.add_patch(bobA2)
    ax.plot([0], [0], marker='o')

    # Péndulo B
    lineB1, = ax.plot([0, 0], [0, -L1.value], lw=2, alpha=0.6)
    lineB2, = ax.plot([0, 0], [-L1.value, -(L1.value+L2.value)], lw=2, alpha=0.6)
    bobB1 = plt.Circle((0, -L1.value), radius=0.05*L1.value*m1.value, alpha=0.5)
    bobB2 = plt.Circle((0, -(L1.value+L2.value)), radius=0.05*L2.value*m2.value, alpha=0.5)
    ax.add_patch(bobB1); ax.add_patch(bobB2)

    title = ax.set_title("")

    # ------------------ Trayectorias precomputadas ----------------------------
    th1_path_A = th2_path_A = w1_path_A = w2_path_A = None
    th1_path_B = th2_path_B = w1_path_B = w2_path_B = None

    def params_B():
        return (
            L1.value*(1+eps_L1.value),
            L2.value*(1+eps_L2.value),
            m1.value*(1+eps_m1.value),
            m2.value*(1+eps_m2.value),
            g.value *(1+eps_g.value)
        )

    def ic_B_efectivas():
        # IC de B multiplicadas por (1+epsilon_*B)
        return (
            th1B.value*(1+eps_th1B.value),
            th2B.value*(1+eps_th2B.value),
            w1B.value *(1+eps_w1B.value),
            w2B.value *(1+eps_w2B.value)
        )

    def recompute_trayectorias(_=None):
        nonlocal th1_path_A, th2_path_A, w1_path_A, w2_path_A
        nonlocal th1_path_B, th2_path_B, w1_path_B, w2_path_B

        n_steps = int(play.max)

        th1_path_A, th2_path_A, w1_path_A, w2_path_A = integrar_doble_pendulo_rk4(
            th1A.value, th2A.value, w1A.value, w2A.value,
            L1.value, L2.value, m1.value, m2.value, g.value,
            dt=0.01, n_steps=n_steps
        )

        L1B, L2B, m1B, m2B, gB = params_B()
        th1B_eff, th2B_eff, w1B_eff, w2B_eff = ic_B_efectivas()
        th1_path_B, th2_path_B, w1_path_B, w2_path_B = integrar_doble_pendulo_rk4(
            th1B_eff, th2B_eff, w1B_eff, w2B_eff,
            L1B, L2B, m1B, m2B, gB,
            dt=0.01, n_steps=n_steps
        )

        fijar_limites()
        actualizar()

    def radio(L, m):
        return 0.05*L*m  # tamaño proporcional a la masa

    def actualizar(_=None):
        idx = int(min(max(0, t.value), play.max))

        # --- Péndulo A
        if th1_path_A is None:
            th1A_i, th2A_i, w1A_i, w2A_i = th1A.value, th2A.value, w1A.value, w2A.value
        else:
            th1A_i = float(th1_path_A[idx]); th2A_i = float(th2_path_A[idx])
            w1A_i  = float(w1_path_A[idx]);  w2A_i  = float(w2_path_A[idx])

        x1A = L1.value*math.sin(th1A_i); y1A = -L1.value*math.cos(th1A_i)
        x2A = x1A + L2.value*math.sin(th2A_i); y2A = y1A - L2.value*math.cos(th2A_i)

        lineA1.set_data([0, x1A], [0, y1A])
        lineA2.set_data([x1A, x2A], [y1A, y2A])
        bobA1.center = (x1A, y1A); bobA1.set_radius(radio(L1.value, m1.value))
        bobA2.center = (x2A, y2A); bobA2.set_radius(radio(L2.value, m2.value))

        # --- Péndulo B
        L1B, L2B, m1B, m2B, gB = params_B()
        visibleB = bool(mostrar_B.value)
        lineB1.set_visible(visibleB); lineB2.set_visible(visibleB)
        bobB1.set_visible(visibleB);  bobB2.set_visible(visibleB)

        if visibleB:
            if th1_path_B is None:
                th1B_eff, th2B_eff, w1B_eff, w2B_eff = ic_B_efectivas()
                th1B_i, th2B_i = th1B_eff, th2B_eff
            else:
                th1B_i = float(th1_path_B[idx]); th2B_i = float(th2_path_B[idx])

            x1B = L1B*math.sin(th1B_i); y1B = -L1B*math.cos(th1B_i)
            x2B = x1B + L2B*math.sin(th2B_i); y2B = y1B - L2B*math.cos(th2B_i)

            lineB1.set_data([0, x1B], [0, y1B])
            lineB2.set_data([x1B, x2B], [y1B, y2B])
            bobB1.center = (x1B, y1B); bobB1.set_radius(radio(L1B, m1B))
            bobB2.center = (x2B, y2B); bobB2.set_radius(radio(L2B, m2B))

        title.set_text(f"Péndulo doble — t={idx*dt:.2f} s [RK4]")
        fig.canvas.draw_idle()

    # Observadores
    for w in (L1, L2, m1, m2, g,
              th1A, th2A, w1A, w2A,
              th1B, th2B, w1B, w2B,
              eps_L1, eps_L2, eps_m1, eps_m2, eps_g,
              eps_th1B, eps_th2B, eps_w1B, eps_w2B):
        w.observe(recompute_trayectorias, names='value')
    for w in (t, mostrar_B):
        w.observe(actualizar, names='value')

    # Layout
    box_fis    = VBox([L1, L2, m1, m2, g], layout=Layout(width='100%'))
    box_eps_f  = VBox([eps_L1, eps_L2, eps_m1, eps_m2, eps_g], layout=Layout(width='100%'))
    box_A      = VBox([th1A, th2A, w1A, w2A], layout=Layout(width='100%'))
    box_B      = VBox([th1B, th2B, w1B, w2B], layout=Layout(width='100%'))
    box_eps_ic = VBox([eps_th1B, eps_th2B, eps_w1B, eps_w2B], layout=Layout(width='100%'))
    box_time   = VBox([play, t, mostrar_B], layout=Layout(width='100%'))

    controles = VBox([
        box_time,
        box_fis,
        HBox([box_eps_f,box_eps_ic]),
        HBox([box_A, box_B])
    ], layout=Layout(width='45%'))

    display(HBox([controles, fig.canvas]))
    recompute_trayectorias()


# ------------------------------------------------------------------------------
# Ejercicio 6: Gráfica paramétrica de ángulos
# ------------------------------------------------------------------------------

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def grafica_parametrica_angulos(theta1_0, theta2_0, w1_0, w2_0,
                                L1, L2, m1, m2, g,
                                dt=0.01, T=30.0,
                                wrap='mod'):
    """
    Dibuja la curva paramétrica (θ1(t), θ2(t)) coloreada por tiempo.
    wrap: 'mod' -> mapea a (-pi, pi]; 'unwrap' -> desenrolla; None -> sin tocar.
    """
    n_steps = int(T/dt)
    th1, th2, w1, w2 = integrar_doble_pendulo_rk4(theta1_0, theta2_0, w1_0, w2_0,
                                                   L1, L2, m1, m2, g, dt, n_steps)
    t = np.arange(n_steps+1)*dt

    if wrap == 'mod':
        th1p = wrap_pi(th1)
        th2p = wrap_pi(th2)
    elif wrap == 'unwrap':
        th1p = np.unwrap(th1)
        th2p = np.unwrap(th2)
    else:
        th1p, th2p = th1, th2

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(th1p, th2p, c=t, s=6, cmap='viridis', lw=0, alpha=0.9)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('t (s)')
    ax.set_xlabel(r'$\theta_1$ (rad)')
    ax.set_ylabel(r'$\theta_2$ (rad)')
    ax.set_title(r'Evolución paramétrica de ángulos $(\theta_1,\ \theta_2)$')
    ax.grid(True)
    plt.show()