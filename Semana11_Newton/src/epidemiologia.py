# ============================================
# Librerías base y utilidades
# ============================================
from __future__ import annotations
from typing import Callable, Iterable, List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ajustes de impresión
np.set_printoptions(precision=6, suppress=True)
pd.set_option("display.precision", 6)

# Función auxiliar para asegurar arrays 1D/2D float
def _as_float_array(x) -> np.ndarray:
    """
    Convierte la entrada en arreglo numpy de tipo float (copia segura).

    Firma
    -----
    def _as_float_array(x) -> np.ndarray

    Parámetros
    ----------
    x : objeto convertible a array

    Retorna
    -------
    np.ndarray
        Arreglo de floats (al menos 1D).
    """
    return np.array(x, dtype=float, copy=True)
# ============================================
def newton_1d(
    funcion: Callable[[float], float],
    derivada: Callable[[float], float],
    x0: float,
    tolerancia: float = 1e-12,
    max_iteraciones: int = 50
) -> Tuple[float, int, List[float]]:
    """
    Método de Newton en una dimensión.

    Firma
    -----
    def newton_1d(funcion: Callable[[float], float],
                  derivada: Callable[[float], float],
                  x0: float,
                  tolerancia: float = 1e-12,
                  max_iteraciones: int = 50) -> Tuple[float, int, List[float]]

    Parámetros
    ----------
    funcion : Callable[[float], float]
        Función escalar f(x) cuyo cero se desea encontrar.
    derivada : Callable[[float], float]
        Derivada f'(x).
    x0 : float
        Aproximación inicial.
    tolerancia : float, opcional
        Criterio de paro relativo para el incremento (por defecto 1e-12).
    max_iteraciones : int, opcional
        Máximo número de iteraciones (por defecto 50).

    Retorna
    -------
    Tuple[float, int, List[float]]
        (raiz, iteraciones_usadas, historial_de_iterandos)

    Lanza
    -----
    ZeroDivisionError
        Si la derivada se anula en algún paso.
    RuntimeError
        Si no se alcanza la tolerancia en max_iteraciones.
    """
    x = float(x0)
    historial = [x]
    for k in range(1, max_iteraciones + 1):
        fx = funcion(x)
        dfx = derivada(x)
        if dfx == 0.0:
            raise ZeroDivisionError("Derivada nula durante Newton 1D.")
        incremento = fx / dfx
        x_nuevo = x - incremento
        historial.append(float(x_nuevo))
        # Criterio de paro relativo en el incremento
        if abs(x_nuevo - x) <= tolerancia * max(1.0, abs(x_nuevo)):
            return x_nuevo, k, historial
        x = x_nuevo
    raise RuntimeError("Newton 1D no convergió dentro del número máximo de iteraciones.")

# ============================================
def trapecio_implicito_1d(
    f: Callable[[float, float], float],
    df_dy: Callable[[float, float], float],
    t0: float,
    y0: float,
    h: float,
    n_pasos: int,
    tolerancia_newton: float = 1e-12,
    max_iter_newton: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Método del trapecio implícito (RK2) para una EDO escalar y'(t)=f(t,y).

    Firma
    -----
    def trapecio_implicito_1d(f, df_dy, t0, y0, h, n_pasos,
                              tolerancia_newton=1e-12, max_iter_newton=30) -> (t, y)

    Parámetros
    ----------
    f : Callable[[float, float], float]
        Campo de velocidades f(t,y).
    df_dy : Callable[[float, float], float]
        Derivada parcial ∂f/∂y evaluable en (t,y).
    t0 : float
        Tiempo inicial.
    y0 : float
        Valor inicial y(t0)=y0.
    h : float
        Tamaño de paso.
    n_pasos : int
        Número de pasos a realizar.
    tolerancia_newton : float
        Tolerancia para Newton interno.
    max_iter_newton : int
        Máximo de iteraciones de Newton por paso.

    Retorna
    -------
    Tuple[np.ndarray, np.ndarray]
        Vectores (t, y) con los nodos y aproximaciones.
    """
    t = np.linspace(t0, t0 + n_pasos*h, n_pasos + 1, dtype=float)
    y = np.empty(n_pasos + 1, dtype=float)
    y[0] = y0

    for n in range(n_pasos):
        tn, yn = t[n], y[n]
        tn1 = t[n+1]
        fn = f(tn, yn)

        # Definimos g(z) y su derivada para el paso (n -> n+1)
        def g(z: float) -> float:
            return z - yn - 0.5*h*(fn + f(tn1, z))

        def dg(z: float) -> float:
            return 1.0 - 0.5*h*df_dy(tn1, z)

        # Predicción tipo Euler explícito
        z = yn + h*fn

        # Newton 1D para resolver g(z)=0
        for _ in range(max_iter_newton):
            gz = g(z)
            dgz = dg(z)
            if dgz == 0.0:
                raise ZeroDivisionError("Derivada nula en Newton interno (trapecio 1D).")
            z_nuevo = z - gz/dgz
            if abs(z_nuevo - z) <= tolerancia_newton*max(1.0, abs(z_nuevo)):
                z = z_nuevo
                break
            z = z_nuevo
        else:
            raise RuntimeError("Newton interno no convergió en el método del trapecio (1D).")

        y[n+1] = z

    return t, y
# ============================================
def euler_implicito_1d(
    f: Callable[[float, float], float],
    df_dy: Callable[[float, float], float],
    t0: float,
    y0: float,
    h: float,
    n_pasos: int,
    tolerancia_newton: float = 1e-12,
    max_iter_newton: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Método **Euler implícito** (para comparar) en una EDO escalar y'(t)=f(t,y).

    Firma
    -----
    def euler_implicito_1d(f, df_dy, t0, y0, h, n_pasos,
                           tolerancia_newton=1e-12, max_iter_newton=30) -> (t, y)

    Ecuación implícita por paso:
        y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})

    Se resuelve g(z)= z - y_n - h f(t_{n+1}, z) = 0 con Newton.
    """
    t = np.linspace(t0, t0 + n_pasos*h, n_pasos + 1, dtype=float)
    y = np.empty(n_pasos + 1, dtype=float)
    y[0] = y0

    for n in range(n_pasos):
        tn1 = t[n+1]
        yn = y[n]

        def g(z: float) -> float:
            return z - yn - h*f(tn1, z)

        def dg(z: float) -> float:
            return 1.0 - h*df_dy(tn1, z)

        # Predicción tipo Euler explícito (usando f en t_n, y_n)
        z = yn + h*f(t[n], yn)

        for _ in range(max_iter_newton):
            gz, dgz = g(z), dg(z)
            if dgz == 0.0:
                raise ZeroDivisionError("Derivada nula en Newton interno (Euler implícito 1D).")
            z_nuevo = z - gz/dgz
            if abs(z_nuevo - z) <= tolerancia_newton*max(1.0, abs(z_nuevo)):
                z = z_nuevo
                break
            z = z_nuevo
        else:
            raise RuntimeError("Newton interno no convergió en Euler implícito (1D).")

        y[n+1] = z
    return t, y
# ============================================
# Dinámica SI en 1D (variable I)
def f_SI(t: float, I: float, beta: float, N: float) -> float:
    """
    Campo f(t,I) para el modelo SI reducido a 1D:
        I' = beta * I * (1 - I/N)

    Firma
    -----
    def f_SI(t: float, I: float, beta: float, N: float) -> float
    """
    return beta * I * (1.0 - I / N)

def df_dI_SI(t: float, I: float, beta: float, N: float) -> float:
    """
    Derivada parcial respecto de I del campo SI:
        d/dI [ beta * I * (1 - I/N) ] = beta * (1 - 2I/N)

    Firma
    -----
    def df_dI_SI(t: float, I: float, beta: float, N: float) -> float
    """
    return beta * (1.0 - 2.0 * I / N)

def I_SI_explicita(t: np.ndarray, beta: float, N: float, I0: float) -> np.ndarray:
    """
    Solución explícita de la logística (modelo SI reducido):
        I(t) = N / ( 1 + ((N/I0)-1) * exp(-beta*t) )

    Firma
    -----
    def I_SI_explicita(t: np.ndarray, beta: float, N: float, I0: float) -> np.ndarray
    """
    t = _as_float_array(t)
    c = (N / I0) - 1.0
    return N / (1.0 + c * np.exp(-beta * t))
# ============================================
def error_global_infinito_1d(
    t: np.ndarray,
    y_num: np.ndarray,
    solucion_exacta: Callable[[np.ndarray], np.ndarray]
) -> float:
    """
    Error global (norma infinito discreta) en 1D:
        E = max_n | y_n - y(t_n) |

    Firma
    -----
    def error_global_infinito_1d(t, y_num, solucion_exacta) -> float
    """
    y_ex = solucion_exacta(t)
    return float(np.max(np.abs(y_num - y_ex)))

def construir_tabla_eoc(hs: List[float], errs: List[float]) -> pd.DataFrame:
    """
    Construye una tabla con columnas: (h, err_max, err_rate) donde
    err_rate es la EOC respecto al tamaño de paso previo (NaN en la primera fila).

    Firma
    -----
    def construir_tabla_eoc(hs: List[float], errs: List[float]) -> pd.DataFrame
    """
    datos = []
    for k, (h, e) in enumerate(zip(hs, errs)):
        if k == 0:
            rate = np.nan
        else:
            rate = np.log(errs[k-1]/e) / np.log(hs[k-1]/h)
        datos.append((h, e, rate))
    df = pd.DataFrame(datos, columns=["h", "err_max", "err_rate"])    
    return df
# ============================================
def grafica_error_vs_h_loglog(hs: Iterable[float], errores: Iterable[float], titulo: str) -> None:
    """
    Dibuja loglog(h, error). Se evita usar estilos o colores específicos.

    Firma
    -----
    def grafica_error_vs_h_loglog(hs: Iterable[float], errores: Iterable[float], titulo: str) -> None
    """
    hs = np.array(list(hs), dtype=float)
    errores = np.array(list(errores), dtype=float)
    plt.figure()
    plt.loglog(hs, errores, marker='o')
    plt.xlabel('h')
    plt.ylabel('err_max (norma infinito)')
    plt.title(titulo)
    plt.grid(True, which='both')
    plt.show()
# ============================================
def F_SIR(t: float, y: np.ndarray, beta: float, gamma: float, N: float) -> np.ndarray:
    """
    Campo del sistema SIR reducido a (S, I).

    Firma
    -----
    def F_SIR(t: float, y: np.ndarray, beta: float, gamma: float, N: float) -> np.ndarray
    """
    S, I = float(y[0]), float(y[1])
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    return np.array([dS, dI], dtype=float)

def JF_SIR(t: float, y: np.ndarray, beta: float, gamma: float, N: float) -> np.ndarray:
    """
    Jacobiano del campo SIR respecto de y=(S,I).

    Firma
    -----
    def JF_SIR(t: float, y: np.ndarray, beta: float, gamma: float, N: float) -> np.ndarray
    """
    S, I = float(y[0]), float(y[1])
    dS_dS = -beta * I / N
    dS_dI = -beta * S / N
    dI_dS =  beta * I / N
    dI_dI =  beta * S / N - gamma
    return np.array([[dS_dS, dS_dI],
                     [dI_dS, dI_dI]], dtype=float)
# ============================================
def newton_sistema(
    G: Callable[[np.ndarray], np.ndarray],
    JG: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tolerancia: float = 1e-12,
    max_iteraciones: int = 50
) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """
    Método de Newton para sistemas G(x)=0.

    Firma
    -----
    def newton_sistema(G, JG, x0, tolerancia=1e-12, max_iteraciones=50)
        -> Tuple[np.ndarray, int, List[np.ndarray]]

    Parámetros
    ----------
    G : Callable[[np.ndarray], np.ndarray]
        Función vectorial.
    JG : Callable[[np.ndarray], np.ndarray]
        Jacobiano de G.
    x0 : np.ndarray
        Estimación inicial (vector).
    tolerancia : float
        Tolerancia relativa para el incremento.
    max_iteraciones : int
        Máximo número de iteraciones.

    Retorna
    -------
    (x, iteraciones, historial)
        x : aproximación a la raíz
        iteraciones : número de iteraciones usadas
        historial : lista de iterandos
    """
    x = _as_float_array(x0)
    historial = [x.copy()]
    for k in range(1, max_iteraciones + 1):
        Gx = _as_float_array(G(x))
        J = _as_float_array(JG(x))
        # Resuelve J s = Gx
        s = np.linalg.solve(J, Gx)
        x_nuevo = x - s
        historial.append(x_nuevo.copy())
        if np.linalg.norm(x_nuevo - x, ord=np.inf) <= tolerancia * max(1.0, np.linalg.norm(x_nuevo, ord=np.inf)):
            return x_nuevo, k, historial
        x = x_nuevo
    raise RuntimeError("Newton (sistema) no convergió en el máximo de iteraciones.")
# ============================================
def trapecio_implicito_sistema(
    F: Callable[[float, np.ndarray], np.ndarray],
    JF: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    y0: np.ndarray,
    h: float,
    n_pasos: int,
    tolerancia_newton: float = 1e-12,
    max_iter_newton: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Método del trapecio (Crank–Nicolson) para sistemas y'(t)=F(t,y).

    Firma
    -----
    def trapecio_implicito_sistema(F, JF, t0, y0, h, n_pasos,
                                   tolerancia_newton=1e-12, max_iter_newton=30) -> (t, Y)

    En cada paso se resuelve:
        G(z) = z - y_n - (h/2) * ( F(t_n, y_n) + F(t_{n+1}, z) ) = 0
    con
        JG(z) = I - (h/2) * JF(t_{n+1}, z).
    """
    y0 = _as_float_array(y0)
    m = y0.size
    t = np.linspace(t0, t0 + n_pasos*h, n_pasos + 1, dtype=float)
    Y = np.empty((n_pasos + 1, m), dtype=float)
    Y[0, :] = y0

    Iden = np.eye(m)

    for n in range(n_pasos):
        tn, yn = t[n], Y[n, :].copy()
        tn1 = t[n+1]
        Fn = _as_float_array(F(tn, yn))

        def G(z: np.ndarray) -> np.ndarray:
            return z - yn - 0.5*h*(Fn + _as_float_array(F(tn1, z)))

        def JG(z: np.ndarray) -> np.ndarray:
            return Iden - 0.5*h*_as_float_array(JF(tn1, z))

        # Predicción de Euler explícito
        z = yn + h*Fn

        # Newton vectorial
        for _ in range(max_iter_newton):
            Gz = G(z)
            Jz = JG(z)
            s = np.linalg.solve(Jz, Gz)
            z_nuevo = z - s
            if np.linalg.norm(z_nuevo - z, ord=np.inf) <= tolerancia_newton*max(1.0, np.linalg.norm(z_nuevo, ord=np.inf)):
                z = z_nuevo
                break
            z = z_nuevo
        else:
            raise RuntimeError("Newton interno no convergió en trapecio (sistema).")

        Y[n+1, :] = z
    return t, Y


def rk4_sistema(
    F: Callable[[float, np.ndarray], np.ndarray],
    t0: float,
    y0: np.ndarray,
    h: float,
    n_pasos: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Método clásico de Runge–Kutta de orden 4 para sistemas.

    Firma
    -----
    def rk4_sistema(F, t0, y0, h, n_pasos) -> (t, Y)
    """
    y0 = _as_float_array(y0)
    m = y0.size
    t = np.linspace(t0, t0 + n_pasos*h, n_pasos + 1, dtype=float)
    Y = np.empty((n_pasos + 1, m), dtype=float)
    Y[0, :] = y0

    for n in range(n_pasos):
        tn, yn = t[n], Y[n, :]
        k1 = _as_float_array(F(tn, yn))
        k2 = _as_float_array(F(tn + 0.5*h, yn + 0.5*h*k1))
        k3 = _as_float_array(F(tn + 0.5*h, yn + 0.5*h*k2))
        k4 = _as_float_array(F(tn + h, yn + h*k3))
        Y[n+1, :] = yn + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
    return t, Y


def error_global_infinito_vector(
    Y_num: np.ndarray,
    Y_ref: np.ndarray
) -> float:
    """
    Error global con norma infinito en el tiempo **y** en los componentes:
        E = max_{n} max_{i} | (Y_num)_{n,i} - (Y_ref)_{n,i} |

    Firma
    -----
    def error_global_infinito_vector(Y_num, Y_ref) -> float
    """
    dif = np.abs(Y_num - Y_ref)
    return float(np.max(dif))
# ============================================
def F_SIRV(t: float, y: np.ndarray, beta: float, gamma: float, nu: float, N: float) -> np.ndarray:
    """
    Campo del sistema SIRV en (S, I, R, V).

    Firma
    -----
    def F_SIRV(t: float, y: np.ndarray, beta: float, gamma: float, nu: float, N: float) -> np.ndarray
    """
    S, I, R, V = _as_float_array(y)
    dS = -beta * S * I / N - nu * S
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    dV =  nu * S
    return np.array([dS, dI, dR, dV], dtype=float)

def JF_SIRV(t: float, y: np.ndarray, beta: float, gamma: float, nu: float, N: float) -> np.ndarray:
    """
    Jacobiano del campo SIRV respecto de y=(S,I,R,V).

    Firma
    -----
    def JF_SIRV(t: float, y: np.ndarray, beta: float, gamma: float, nu: float, N: float) -> np.ndarray
    """
    S, I, R, V = _as_float_array(y)
    J = np.zeros((4,4), dtype=float)
    # dS/dS, dS/dI
    J[0,0] = -beta * I / N - nu
    J[0,1] = -beta * S / N
    # dI/dS, dI/dI
    J[1,0] =  beta * I / N
    J[1,1] =  beta * S / N - gamma
    # dR/dI
    J[2,1] =  gamma
    # dV/dS
    J[3,0] =  nu
    return J
# ============================================
