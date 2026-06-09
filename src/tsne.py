import numpy as np

def _pairwise_squared_distances(X):
    """
    Calcula la matriz de distancias euclídeas al cuadrado entre todos los pares
    de puntos de X.

    """
    sum_sq = np.sum(X ** 2, axis=1) 
    D = sum_sq[:, None] + sum_sq[None, :] - 2 * (X @ X.T)
    np.fill_diagonal(D, 0.0)  
    D = np.maximum(D, 0.0)         
    return D


def _binary_search_sigma(dist2_i, target_perplexity, max_iter=200, tol=1e-5):
    """
    Encuentra el sigma_i para el punto i mediante búsqueda binaria, de modo que
    la perplejidad de la distribución condicional p_{j|i} sea igual a
    target_perplexity.
    """
    beta_min = -np.inf   # beta = 1 / (2 * sigma^2)
    beta_max = np.inf
    beta =  1.0

    for _ in range(max_iter):
        # Calcular p_{j|i} con el beta actual
        exp_d = np.exp(-dist2_i * beta)
        sum_exp = np.sum(exp_d)

        if sum_exp == 0:
            sum_exp = 1e-10

        p_i = exp_d / sum_exp
        H = -np.sum(p_i * np.log2(np.maximum(p_i, 1e-10)))

        # Perplejidad actual
        perp = 2.0 ** H

        # Ajustar beta según si la perplejidad es mayor o menor a la deseada
        diff = perp - target_perplexity
        if np.abs(diff) < tol:
            break

        if diff > 0:
            # Perplejidad demasiado alta → aumentar beta (reducir sigma)
            beta_min = beta
            beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
        else:
            # Perplejidad demasiado baja → reducir beta (aumentar sigma)
            beta_max = beta
            beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

    return p_i


def _compute_joint_probabilities(X, perplexity):
    """
    Calcula la matriz de probabilidades conjuntas simetrizadas P de shape (n, n).
    """
    n = X.shape[0]
    D = _pairwise_squared_distances(X)

    # Matriz de probabilidades condicionales
    P_cond = np.zeros((n, n))
    for i in range(n):
        dist2_i = np.concatenate([D[i, :i], D[i, i+1:]])
        p_i     = _binary_search_sigma(dist2_i, perplexity)

        P_cond[i, :i]   = p_i[:i]
        P_cond[i, i+1:] = p_i[i:]

    # Simetrizar y normalizar
    P = (P_cond + P_cond.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)

    return P


def _compute_q_matrix(Y):
    """
    Calcula la matriz Q en el espacio de baja dimensión usando una distribución
    t de Student con 1 grado de libertad
    """
    n = Y.shape[0]
    D = _pairwise_squared_distances(Y) 
    num = 1.0 / (1.0 + D)       
    np.fill_diagonal(num, 0.0)         

    sum_num = np.sum(num)
    Q = num / sum_num
    Q = np.maximum(Q, 1e-12)

    return Q, num


def _compute_gradient(P, Q, Y, num):
    """
    Calcula el gradiente de la divergencia KL(P||Q) respecto a Y.

    """
    n = Y.shape[0]
    PQ = P - Q    
    PQ_num = PQ * num        
    diff = Y[:, None, :] - Y[None, :, :]  
    grad = 4.0 * np.einsum('ij,ijk->ik', PQ_num, diff)  

    return grad


def tsne(X, n_components=2, perplexity=30, n_iter=1000, lr=200.0,
         early_exaggeration=4.0, exaggeration_iter=250,
         momentum_init=0.5, momentum_final=0.8,
         random_seed=42, verbose=True):
    """
    Implementación propia de t-SNE (t-distributed Stochastic Neighbor Embedding).

    Reduce X de alta dimensión a n_components dimensiones preservando
    la estructura local de vecindades.
    """
    np.random.seed(random_seed)
    n = X.shape[0]

    if verbose:
        print(f"[t-SNE] n={n}, d={X.shape[1]}, perplexity={perplexity}, "
              f"n_iter={n_iter}, lr={lr}")

    # Paso 1: calcular P (probabilidades conjuntas en alta dimensión)
    P = _compute_joint_probabilities(X, perplexity)

    # Paso 2: inicializar embedding Y con ruido gaussiano pequeño
    Y  = np.random.randn(n, n_components) * 0.01
    Y_prev  = Y.copy()     # para momentum
    update  = np.zeros_like(Y)

    if verbose:
        print("[t-SNE] Iniciando optimización...")

    # Paso 3: gradient descent con momentum y early exaggeration
    for t in range(1, n_iter + 1):

        # Early exaggeration: en las primeras iteraciones multiplicar P
        # para que los clusters se separen más rápido
        if t <= exaggeration_iter:
            P_use    = P * early_exaggeration
            momentum = momentum_init
        else:
            P_use    = P
            momentum = momentum_final

        # Calcular Q y gradiente
        Q, num = _compute_q_matrix(Y)
        grad = _compute_gradient(P_use, Q, Y, num)

        # Actualizar con momentum:
        update = momentum * update - lr * grad
        Y = Y + update

        # Centrar Y en cada iteración para estabilidad numérica
        Y -= Y.mean(axis=0)

        if verbose and t % 100 == 0:
            # Calcular KL divergence
            kl = np.sum(P * np.log(np.maximum(P, 1e-12) /
                                   np.maximum(Q, 1e-12)))
            print(f"[t-SNE] Iter {t:4d}/{n_iter} | KL divergence: {kl:.4f}")


    return Y