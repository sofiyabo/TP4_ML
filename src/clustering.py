import numpy as np

def kmeans(X, K, max_iter=300, tol=1e-4, random_seed=42, n_init=10):
    """
    Corre K-Means n_init veces y devuelve el resultado con menor inercia.
    """
    best_labels, best_centroids, best_inertia = None, None, np.inf

    for run in range(n_init):
        rng = np.random.default_rng(random_seed + run)

        first = rng.integers(0, len(X))
        centroids = [X[first]]
        for _ in range(K - 1):
            C = np.array(centroids) 
            D2 = np.min(np.sum((X[:, None] - C[None, :]) ** 2, axis=2), axis=1)
            probs = D2 / D2.sum()
            centroids.append(X[rng.choice(len(X), p=probs)])
        centroids = np.array(centroids)

        for _ in range(max_iter):
            dists  = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
                for k in range(K)
            ])
            if np.linalg.norm(new_centroids - centroids) < tol:
                break
            centroids = new_centroids

        inertia = sum(
            (np.linalg.norm(X[labels == k] - centroids[k], axis=1) ** 2).sum()
            for k in range(K)
        )
        if inertia < best_inertia:
            best_labels, best_centroids, best_inertia = labels.copy(), centroids.copy(), inertia

    return best_labels, best_centroids, best_inertia


def gmm(X, K, max_iter=300, tol=1e-4, random_seed=42, reg_covar=1e-2):
    """
    Algoritmo GMM inicializado con KMeans.
    """
    np.random.seed(random_seed)
    N, D = X.shape

    labels_init, centroids_init, _ = kmeans(X, K, random_seed=random_seed)

    pi    = np.array([np.mean(labels_init == k) for k in range(K)])
    mu    = centroids_init.copy()
    sigma = np.array([
        np.var(X[labels_init == k], axis=0) + reg_covar   # <-- reg_covar mayor
        for k in range(K)
    ])

    log_likelihood = -np.inf

    for _ in range(max_iter):
        # E-step
        log_resp = np.zeros((N, K))
        for k in range(K):
            diff    = X - mu[k]
            log_det = np.sum(np.log(sigma[k]))
            mahal   = np.sum(diff ** 2 / sigma[k], axis=1)
            log_resp[:, k] = (np.log(pi[k] + 1e-300)
                              - 0.5 * (log_det + mahal + D * np.log(2 * np.pi)))

        log_resp_max = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - log_resp_max)
        resp /= resp.sum(axis=1, keepdims=True)

        # M-step
        Nk = resp.sum(axis=0)
        # Reinicializar clusters colapsados (Nk muy bajo)
        for k in range(K):
            if Nk[k] < 1.0:
                rand_idx   = np.random.randint(N)
                mu[k]      = X[rand_idx]
                sigma[k]   = np.var(X, axis=0) + reg_covar
                pi[k]      = 1.0 / K
                Nk[k]      = 1.0

        pi    = Nk / N
        mu    = (resp.T @ X) / Nk[:, None]
        sigma = np.array([
            np.sum(resp[:, k:k+1] * (X - mu[k]) ** 2, axis=0) / Nk[k] + reg_covar
            for k in range(K)
        ])

        # Log-likelihood
        for k in range(K):
            diff    = X - mu[k]
            log_det = np.sum(np.log(sigma[k]))
            mahal   = np.sum(diff ** 2 / sigma[k], axis=1)

        lrm = log_resp.max(axis=1, keepdims=True)
        new_ll = np.sum(lrm.squeeze() + np.log(np.sum(np.exp(log_resp - lrm), axis=1)))

        if abs(new_ll - log_likelihood) < tol:
            log_likelihood = new_ll
            break
        log_likelihood = new_ll

    labels = np.argmax(resp, axis=1)
    return labels, mu, sigma, pi, log_likelihood

def silhouette_score(X, labels):
    N = len(X)
    clusters = np.unique(labels)

    # Matriz de distancias
    sq = (X ** 2).sum(axis=1)   
    dist = np.sqrt(
        np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    )           

    a = np.zeros(N)
    b = np.full(N, np.inf)

    for k in clusters:
        mask = labels == k
        idx  = np.where(mask)[0]
        size = idx.shape[0]

        # a(i) para los puntos del cluster k — vectorizado
        if size > 1:
            intra = dist[np.ix_(idx, idx)]  
            a[idx] = intra.sum(axis=1) / (size - 1)
        # else a[i] queda 0

        # b(i): distancia promedio a cada otro cluster
        for other in clusters:
            if other == k:
                continue
            other_idx = np.where(labels == other)[0]
            mean_inter = dist[np.ix_(idx, other_idx)].mean(axis=1) 
            b[idx] = np.minimum(b[idx], mean_inter)

    s = (b - a) / np.maximum(a, b)
    s[np.isnan(s)] = 0.0
    return s.mean()