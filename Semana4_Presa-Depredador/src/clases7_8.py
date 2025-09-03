import numpy as np
import matplotlib.pyplot as plt

# 2.1) Definición del sistema presa–depredador (Lotka–Volterra)
def presa_depredador(t, z, params):
    """
    Sistema presa–depredador de Lotka–Volterra.
    
    Parámetros
    ----------
    t : float
        Tiempo (no usado explícitamente en este sistema autónomo, pero se incluye por compatibilidad).
    z : array_like, shape (2,)
        Vector de estado [x, y] con x = presas, y = depredadores.
    params : dict
        Diccionario con claves 'alpha', 'beta', 'gamma', 'delta' (todas > 0).
    
    Retorna
    -------
    f : ndarray, shape (2,)
        Derivadas [dx/dt, dy/dt].
    """
    x, y = z
    alpha = params["alpha"]
    beta  = params["beta"]
    gamma = params["gamma"]
    delta = params["delta"]
    dx = x*(alpha - beta*y)
    dy = y*(-gamma + delta*x)
    return np.array([dx, dy], dtype=float)


def jacobiano_presa_depredador(z, params):
    """
    Jacobiano analítico del sistema en el punto z = [x, y].
    """
    x, y = z
    alpha = params["alpha"]
    beta  = params["beta"]
    gamma = params["gamma"]
    delta = params["delta"]
    J11 = alpha - beta*y
    J12 = -beta*x
    J21 =  delta*y
    J22 = -gamma + delta*x
    return np.array([[J11, J12],
                     [J21, J22]], dtype=float)


# 2.1) Integradores generales

def euler_explicito(f, t0, tf, z0, h, params):
    """
    Euler explícito para sistemas z' = f(t, z).
    Retorna tiempos y trayectorias.
    """
    N = int(np.ceil((tf - t0)/h))
    t = t0 + np.arange(N+1)*h
    t[-1] = tf  # forzar fin exacto
    z = np.empty((N+1, len(z0)), dtype=float)
    z[0] = np.array(z0, dtype=float)
    for n in range(N):
        z[n+1] = z[n] + (t[n+1]-t[n])*f(t[n], z[n], params)
    return t, z


def _punto_fijo(update_fn, z_init, max_iter=20, tol=1e-10):
    z = z_init.copy()
    for _ in range(max_iter):
        z_next = update_fn(z)
        if np.linalg.norm(z_next - z, ord=np.inf) < tol:
            return z_next, True
        z = z_next
    return z, False  


def euler_implicito_pc(f, t0, tf, z0, h, params, max_iter=30, tol=1e-10):
    """
    Euler implícito con iteración de punto fijo (predictor-corrector).
    """
    N = int(np.ceil((tf - t0)/h))
    t = t0 + np.arange(N+1)*h
    t[-1] = tf
    z = np.empty((N+1, len(z0)), dtype=float)
    z[0] = np.array(z0, dtype=float)
    convergencias = []
    for n in range(N):
        dt = t[n+1]-t[n]
        # Predictor explícito
        z_pred = z[n] + dt*f(t[n], z[n], params)
        # Corrector: z = z_n + dt f(t_{n+1}, z)
        def upd(zk):
            return z[n] + dt*f(t[n+1], zk, params)
        z_np1, ok = _punto_fijo(upd, z_pred, max_iter=max_iter, tol=tol)
        z[n+1] = z_np1
        convergencias.append(ok)
    return t, z, np.array(convergencias, dtype=bool)


def trapecio_pc(f, t0, tf, z0, h, params, max_iter=30, tol=1e-10):
    """
    Método del trapecio con predictor-corrector de punto fijo.
    """
    N = int(np.ceil((tf - t0)/h))
    t = t0 + np.arange(N+1)*h
    t[-1] = tf
    z = np.empty((N+1, len(z0)), dtype=float)
    z[0] = np.array(z0, dtype=float)
    convergencias = []
    for n in range(N):
        dt = t[n+1]-t[n]
        fn = f(t[n], z[n], params)
        # Predictor explícito
        z_pred = z[n] + dt*fn
        # Corrector: z = z_n + dt/2 [ f(t_n, z_n) + f(t_{n+1}, z) ]
        def upd(zk):
            return z[n] + 0.5*dt*(fn + f(t[n+1], zk, params))
        z_np1, ok = _punto_fijo(upd, z_pred, max_iter=max_iter, tol=tol)
        z[n+1] = z_np1
        convergencias.append(ok)
    return t, z, np.array(convergencias, dtype=bool)



def H_invariante(z, params):
    """
    Invariante clásico de Lotka–Volterra (válido para x>0, y>0).
    """
    x = z[:, 0]
    y = z[:, 1]
    alpha = params["alpha"]
    beta  = params["beta"]
    gamma = params["gamma"]
    delta = params["delta"]
    return delta*x - gamma*np.log(x) + beta*y - alpha*np.log(y)



# 2.2) Utilidades de graficación (una figura por celda, sin estilos/colores fijados)

def graficar_tiempo(t, z, etiquetas=("x(t) presas", "y(t) depredadores"), titulo="Series de tiempo"):
    plt.figure()
    plt.plot(t, z[:,0], label=etiquetas[0])
    plt.plot(t, z[:,1], label=etiquetas[1])
    plt.xlabel("t")
    plt.ylabel("población")
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show()


def graficar_fase(z, titulo="Plano fase (x vs y)"):
    plt.figure()
    plt.plot(z[:,0], z[:,1])
    plt.xlabel("x (presas)")
    plt.ylabel("y (depredadores)")
    plt.title(titulo)
    plt.grid(True)
    plt.show()


def graficar_invariante(t, H, titulo="Evolución de H(x,y)"):
    plt.figure()
    plt.plot(t, H)
    plt.xlabel("t")
    plt.ylabel("H(x,y)")
    plt.title(titulo)
    plt.grid(True)
    plt.show()
