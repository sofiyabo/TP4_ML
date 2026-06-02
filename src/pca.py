import numpy as np


def pca_fit(X_std, n_components=None):
    """
    Aprende la transformación PCA sobre X_std (ya estandarizado).
    Mantiene float32 para evitar picos de memoria.
    """
    N = X_std.shape[0]
    
    # Calcular covarianza manualmente en float32 
    X32 = X_std.astype(np.float32)
    mean = X32.mean(axis=0)
    X_centered = X32 - mean                        # (N, 784) float32
    cov = (X_centered.T @ X_centered) / (N - 1)   # (784, 784) float32
    del X_centered                                 # liberar inmediatamente
    
    # eigh para matrices simétricas
    eigenvalues, eigenvectors = np.linalg.eigh(cov.astype(np.float64)) 
    
    # Ordenar de mayor a menor
    orden = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[orden]
    eigenvectors = eigenvectors[:, orden].astype(np.float32)
    
    if n_components is not None:
        eigenvalues  = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
    
    return eigenvalues, eigenvectors

def pca_transform(X_std, eigenvectors):
    """
    Proyecta los datos sobre los componentes principales.
    """
    return X_std @ eigenvectors  # shape: (n_samples, n_components)

def pca_inverse(X_pca, eigenvectors, media, desvio):
    """
    Reconstruye las imágenes desde el espacio PCA al original.
    """
    X_std_rec = X_pca @ eigenvectors.T
    return X_std_rec * desvio + media  # des-estandarizar