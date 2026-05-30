import numpy as np


def pca_fit(X_std, n_components=None):
    """
    Aprende la transformación PCA sobre X_std (ya estandarizado).
    """
    # Matriz de covarianza
    cov = np.cov(X_std.T)  # shape: (784, 784)
    
    # Autovalores y autovectores
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Ordenar de mayor a menor varianza
    orden = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[orden]
    eigenvectors = eigenvectors[:, orden]
    
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