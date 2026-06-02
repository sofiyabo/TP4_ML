import numpy as np

def kmeans(X, K, max_iter=300, tol=1e-4, random_seed=42):
    np.random.seed(random_seed)
    
    # Inicializar centroides aleatoriamente
    idx = np.random.choice(len(X), K, replace=False)
    centroids = X[idx].copy()
    
    for _ in range(max_iter):
        # Asignar cada punto al centroide mas cercano
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)  # (N, K)
        labels = np.argmin(dists, axis=1)
        
        # Actualizar centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    
    inertia = sum(np.linalg.norm(X[labels == k] - centroids[k], axis=1).sum() for k in range(K)) #quiero la inercia mas baja posible
    return labels, centroids, inertia

def gmm(X, K, max_iter=100, tol=1e-4, random_seed=42):
    np.random.seed(random_seed)
    N, D = X.shape

    # Inicializar con k-means
    labels_init, centroids_init, _ = kmeans(X, K, random_seed=random_seed)
    
    # Parámetros iniciales
    pi    = np.array([np.mean(labels_init == k) for k in range(K)])               # (K,)
    mu    = centroids_init.copy()                                                   # (K, D)
    sigma = np.array([np.var(X[labels_init == k], axis=0) + 1e-6 for k in range(K)])  # (K, D)

    log_likelihood = -np.inf

    for _ in range(max_iter):
        # E-step: calcular responsabilidades
        log_resp = np.zeros((N, K))
        for k in range(K):
            diff    = X - mu[k]
            log_det = np.sum(np.log(sigma[k]))
            mahal   = np.sum(diff**2 / sigma[k], axis=1)
            log_resp[:, k] = np.log(pi[k] + 1e-300) - 0.5 * (log_det + mahal + D * np.log(2 * np.pi))

        # Normalizar responsabilidades (log-sum-exp trick)
        log_resp_max = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - log_resp_max)
        resp /= resp.sum(axis=1, keepdims=True)

        # M-step: actualizar parámetros
        Nk = resp.sum(axis=0)                                                      # (K,)
        pi = Nk / N
        mu = (resp.T @ X) / Nk[:, None]                                            # (K, D)
        sigma = np.array([
            np.sum(resp[:, k:k+1] * (X - mu[k])**2, axis=0) / Nk[k] + 1e-6
            for k in range(K)
        ])

        # Chequear convergencia
        new_log_likelihood = np.sum(log_resp_max.squeeze() + np.log(np.sum(np.exp(log_resp - log_resp_max), axis=1)))
        if abs(new_log_likelihood - log_likelihood) < tol:
            break
        log_likelihood = new_log_likelihood

    labels = np.argmax(resp, axis=1)
    return labels, mu, sigma, pi, log_likelihood

def gmm_best_of_n(X, K, n_runs=5, **kwargs):
    best_ll = -np.inf
    best_result = None
    for seed in range(n_runs):
        result = gmm(X, K, random_seed=seed, **kwargs)
        if result[-1] > best_ll:
            best_ll = result[-1]
            best_result = result
    return best_result