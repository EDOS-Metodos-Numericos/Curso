import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from ipywidgets import interact, FloatSlider
from matplotlib.lines import Line2D

def graficar_logistica(r: float = 1.0):
    """
    Grafica x vs x' para la ecuación logística x' = r x (1-x)
    con flechas del flujo en el eje x.
    """
    x_min, x_max = -0.5, 1.5
    x = np.linspace(x_min, x_max, 600)
    f = lambda xx: r*xx*(1-xx)

    plt.figure(figsize=(6,4))
    plt.plot(x, f(x), label="$x' = r x (1-x)$")
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel("$x$")
    plt.ylabel("$x'$")
    plt.title(f"Ecuación logística: $x' = r x (1-x)$ con r={r}")
    plt.grid(True)

    # Equilibrios: 0 (inestable), 1 (estable)
    plt.plot(0, 0, marker='x', linestyle='None', label="Equilibrio inestable (0)")
    plt.plot(1, 0, marker='o', linestyle='None', label="Equilibrio estable (1)")

    # Flechas de flujo en el eje x
    posiciones = np.linspace(x_min, x_max, 15)
    for xi in posiciones:
        signo = f(xi)
        if signo > 0:
            dx = 0.15
        elif signo < 0:
            dx = -0.15
        else:
            dx = 0
        plt.arrow(xi, 0, dx, 0, head_width=0.05, head_length=0.05,
                  fc='k', ec='k', length_includes_head=True)

    plt.legend()
    plt.tight_layout()
    plt.show()

def dibujar_circulo_semi_relleno(ax, mu0, x0, radio=0.08, lado='derecha'):
    """
    Dibuja un círculo con mitad rellena en (mu0, x0) sobre ax.
    lado: 'derecha' o 'izquierda' (mitad rellena).
    """
    # contorno
    circ = Circle((mu0, x0), radius=radio, edgecolor='k', facecolor='white', linewidth=1.5, zorder=3)
    ax.add_patch(circ)
    # wedge (media luna)
    if lado == 'derecha':
        theta1, theta2 = -90, 90   # rellena lado derecho
    else:
        theta1, theta2 = 90, 270   # rellena lado izquierdo
    wedge = Wedge((mu0, x0), r=radio, theta1=theta1, theta2=theta2, facecolor='k', edgecolor='none', zorder=2)
    ax.add_patch(wedge)

def _flechas_flujo_en_eje(ax, f, x_min, x_max, n=15, dx_base=0.25, head_w=0.4, head_l=0.12):
    xs = np.linspace(x_min, x_max, n)
    for xi in xs:
        s = f(xi)
        if s > 0:
            dx = dx_base
        elif s < 0:
            dx = -dx_base
        else:
            dx = 0
        ax.arrow(xi, 0, dx, 0, head_width=head_w, head_length=head_l,
                 fc='k', ec='k', length_includes_head=True, zorder=3)

def graficar_silla_nodo(mu: float):
    # rangos
    x_min, x_max = -3.0, 3.0
    mu_min, mu_max = -2.0, 2.0

    # --- Figura lado a lado ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

    # (izquierda) x vs x'
    x = np.linspace(x_min, x_max, 600)
    f = lambda xx: mu - xx**2
    ax1.plot(x, f(x), label="$x' = \\mu - x^2$")
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$x'$")
    ax1.set_title(f"Silla–nodo (μ = {mu:.2f})")
    ax1.grid(True)

    # equilibrios
    if mu > 0:
        ax1.plot(np.sqrt(mu), 0, marker='o', linestyle='None')   # estable
        ax1.plot(-np.sqrt(mu), 0, marker='x', linestyle='None')  # inestable
    elif mu == 0:
        ax1.plot(0, 0, marker='o', fillstyle='left', linestyle='None')  # semiestable (marcado en x–x')
    # flechas
    _flechas_flujo_en_eje(ax1, f, x_min, x_max)

    # leyenda manual
    marcadores = [Line2D([0],[0], marker='o', linestyle='None', label='Estable'),
                  Line2D([0],[0], marker='x', linestyle='None', label='Inestable')]
    ax1.legend(handles=[Line2D([0],[0])]+marcadores, labels=["$x'=\\mu-x^2$","Estable","Inestable"], loc='best')

    # (derecha) diagrama de bifurcación x* vs μ
    ax2.set_title("Diagrama de bifurcación")
    ax2.set_xlabel("$\\mu$")
    ax2.set_ylabel("$x^*$")
    ax2.grid(True)
    mus = np.linspace(0, mu_max, 400)
    # ramas (μ≥0)
    ax2.plot(mus,  np.sqrt(mus),  linestyle='-',  linewidth=2, color= "orange")  # estable
    ax2.plot(mus, -np.sqrt(mus),  linestyle='--', linewidth=2, color= "green")  # inestable
    # semiestable en (0,0)
    dibujar_circulo_semi_relleno(ax2, 0.0, 0.0, radio=0.06, lado='derecha')
    # línea vertical móvil
    ax2.axvline(mu, linestyle='-', linewidth=1.5)
    ax2.set_xlim(mu_min, mu_max)
    ax2.set_ylim(-2.1, 2.1)
    plt.tight_layout()

    
def graficar_transcritica(mu: float):
    x_min, x_max = -3.0, 3.0
    mu_min, mu_max = -2.0, 2.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

    # (izquierda) x vs x'
    x = np.linspace(x_min, x_max, 600)
    f = lambda xx: mu*xx - xx**2
    ax1.plot(x, f(x), label="$x' = \\mu x - x^2$")
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$x'$")
    ax1.set_title(f"Transcrítica (μ = {mu:.2f})")
    ax1.grid(True)

    # equilibrios: x=0 y x=μ; estabilidad por f'(x*)=μ-2x*
    deriv_0  = mu
    deriv_mu = -mu
    ax1.plot(0, 0, marker=('o' if deriv_0<0 else 'x' if deriv_0>0 else 'o'),
             linestyle='None', fillstyle=('left' if deriv_0==0 else 'full'))
    ax1.plot(mu, 0, marker=('o' if deriv_mu<0 else 'x' if deriv_mu>0 else 'o'),
             linestyle='None', fillstyle=('left' if deriv_mu==0 else 'full'))
    _flechas_flujo_en_eje(ax1, f, x_min, x_max)

    ax1.legend([Line2D([0],[0])], ["$x'=\\mu x - x^2$"], loc='best')

    # (derecha) diagrama de bifurcación
    ax2.set_title("Diagrama de bifurcación")
    ax2.set_xlabel("$\\mu$")
    ax2.set_ylabel("$x^*$")
    ax2.grid(True)

    mus = np.linspace(mu_min, mu_max, 400)
    # ramas: x*=0 (estable si μ<0, inestable si μ>0)
    ax2.plot(mus[mus<0], 0*mus[mus<0], linestyle='-',  linewidth=2, color = "orange")   # estable 
    ax2.plot(mus[mus>0], 0*mus[mus>0], linestyle='--', linewidth=2, color = "orange")   # inestable  
    # rama: x*=μ (inestable si μ<0, estable si μ>0)
    ax2.plot(mus[mus<0], mus[mus<0], linestyle='--', linewidth=2, color = "green")     # inestable
    ax2.plot(mus[mus>0], mus[mus>0], linestyle='-',  linewidth=2, color = "green")     # estable 
    # semiestable en (0,0)
    dibujar_circulo_semi_relleno(ax2, 0.0, 0.0, radio=0.06, lado='derecha')

    ax2.axvline(mu, linestyle='-', linewidth=1.5)
    ax2.set_xlim(mu_min, mu_max)
    ax2.set_ylim(-2.1, 2.1)
    plt.tight_layout()

def graficar_tridente_supercritica(mu: float):
    x_min, x_max = -3.0, 3.0
    mu_min, mu_max = -2.0, 2.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

    # (izquierda) x vs x'
    x = np.linspace(x_min, x_max, 600)
    f = lambda xx: mu*xx - xx**3
    ax1.plot(x, f(x), label="$x' = \\mu x - x^3$")
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$x'$")
    ax1.set_title(f"Tridente supercrítica (μ = {mu:.2f})")
    ax1.grid(True)

    # equilibrios: x=0 siempre; ±sqrt(μ) si μ>0
    # f'(x*)=μ-3x*^2
    d0 = mu
    ax1.plot(0, 0, marker=('o' if d0<0 else 'x' if d0>0 else 'o'),
            linestyle='None', fillstyle=('full' if d0!=0 else 'full'))  # en μ=0 es estable no hiperbólico
    if mu > 0:
        xs = np.sqrt(mu)
        for xeq in (+xs, -xs):
            ax1.plot(xeq, 0, marker='o', linestyle='None')  # estables
    _flechas_flujo_en_eje(ax1, f, x_min, x_max)
    ax1.legend([Line2D([0],[0])], ["$x'=\\mu x - x^3$"], loc='best')

    # (derecha) diagrama de bifurcación
    ax2.set_title("Diagrama de bifurcación")
    ax2.set_xlabel("$\\mu$")
    ax2.set_ylabel("$x^*$")
    ax2.grid(True)

    mus = np.linspace(mu_min, mu_max, 400)
    # rama x*=0: estable si μ<0, inestable si μ>0
    ax2.plot(mus[mus<0], 0*mus[mus<0], linestyle='-',  linewidth=2, color="orange")   # estable
    ax2.plot(mus[mus>0], 0*mus[mus>0], linestyle='--', linewidth=2, color="orange")   # inestable
    # ramas ±sqrt(μ) (μ≥0), estables
    mup = np.linspace(0, mu_max, 300)
    ax2.plot(mup,  np.sqrt(mup), linestyle='-', linewidth=2, color="green")          # estable
    ax2.plot(mup, -np.sqrt(mup), linestyle='-', linewidth=2, color="red")          # estable
    # en μ=0, x*=0 es estable no hiperbólico (círculo lleno normal)
    ax2.plot(0, 0, marker='o', color='k')

    ax2.axvline(mu, linestyle='-', linewidth=1.5)
    ax2.set_xlim(mu_min, mu_max)
    ax2.set_ylim(-2.1, 2.1)
    plt.tight_layout()

def graficar_tridente_subcritica(mu: float):
    x_min, x_max = -3.0, 3.0
    mu_min, mu_max = -2.0, 2.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

    # (izquierda) x vs x'
    x = np.linspace(x_min, x_max, 600)
    f = lambda xx: mu*xx + xx**3
    ax1.plot(x, f(x), label="$x' = \\mu x + x^3$")
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$x'$")
    ax1.set_title(f"Tridente subcrítica (μ = {mu:.2f})")
    ax1.grid(True)

    # equilibrios: x=0 siempre; ±sqrt(-μ) si μ<0
    d0 = mu
    ax1.plot(0, 0, marker=('o' if d0<0 else 'x' if d0>0 else 'x'),
             linestyle='None', fillstyle=('full' if d0!=0 else 'none'))  # en μ=0 es inestable no hiperbólico
    if mu < 0:
        xs = np.sqrt(-mu)
        for xeq in (+xs, -xs):
            # f'(xeq)=mu+3xeq^2 = -2mu > 0 (μ<0) => inestables
            ax1.plot(xeq, 0, marker='x', linestyle='None')
    _flechas_flujo_en_eje(ax1, f, x_min, x_max)
    ax1.legend([Line2D([0],[0])], ["$x'=\\mu x + x^3$"], loc='best')

    # (derecha) diagrama de bifurcación
    ax2.set_title("Diagrama de bifurcación")
    ax2.set_xlabel("$\\mu$")
    ax2.set_ylabel("$x^*$")
    ax2.grid(True)

    mus = np.linspace(mu_min, mu_max, 400)
    # rama x*=0: estable si μ<0 (solida), inestable si μ>0 (punteada)
    ax2.plot(mus[mus<0], 0*mus[mus<0], linestyle='-',  linewidth=2, color="orange")   # estable
    ax2.plot(mus[mus>0], 0*mus[mus>0], linestyle='--', linewidth=2, color="orange")   # inestable
    # ramas ±sqrt(-μ) (μ≤0), inestables
    mun = np.linspace(mu_min, 0, 300)
    ax2.plot(mun,  np.sqrt(-mun), linestyle='--', linewidth=2, color="green")        # inestable
    ax2.plot(mun, -np.sqrt(-mun), linestyle='--', linewidth=2, color="red")        # inestable
    # en μ=0, x*=0 es inestable no hiperbólico (marcador 'x')
    ax2.plot(0, 0, marker='x', color='k')

    ax2.axvline(mu, linestyle='-', linewidth=1.5)
    ax2.set_xlim(mu_min, mu_max)
    ax2.set_ylim(-2.1, 2.1)
    plt.tight_layout()
    
