import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_images(data, n_images=10, titles=None, n_cols=None):
    if isinstance(data, pd.DataFrame):
        pixel_cols = [c for c in data.columns if c.startswith("pixel_")]
        data = data[pixel_cols].values
    
    if not isinstance(data, list):
        data = [data]
    
    n_cols = n_cols or n_images
    idx = np.random.choice(len(data[0]), n_images, replace=False)
    
    fig, axes = plt.subplots(len(data), n_cols, figsize=(n_cols * 1.5, len(data) * 1.8), squeeze=False)
    
    for row, arr in enumerate(data):
        for col, img in enumerate(arr[idx]):
            axes[row, col].imshow(img.reshape(28, 28), cmap="gray")
            axes[row, col].axis("off")
        if titles and row < len(titles):
            axes[row, 0].set_ylabel(titles[row], fontsize=11, rotation=0, labelpad=60, va="center")
    
    plt.tight_layout()
    plt.show()

def barplot(labels, counts):
    plt.figure(figsize=(10, 4))
    plt.bar(labels, counts, color='steelblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Cantidad de muestras')
    plt.title('Distribución de clases en el dataset')
    plt.tight_layout()
    plt.show()

def plot_samples_by_class(X, y, class_names, n_samples=8, classes_to_show=None):
    if classes_to_show is None:
        classes_to_show = sorted(np.unique(y))[:5]

    n_classes = len(classes_to_show)

    fig = plt.figure(figsize=(n_samples * 2, n_classes * 2.5))
    fig.suptitle('Muestras agrupadas por clase', fontsize=15, fontweight='bold')

    for row, cls in enumerate(classes_to_show):
        idx_class = np.where(y == cls)[0]
        selected = np.random.choice(idx_class, size=n_samples, replace=False)

        for col, idx in enumerate(selected):
            ax = fig.add_subplot(n_classes, n_samples, row * n_samples + col + 1)
            img = X[idx].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.axis('off')

            # Título de clase solo en la primera imagen de cada fila
            if col == 0:
                ax.set_title(class_names[cls], fontsize=11, fontweight='bold',
                             loc='left', pad=4)
                
    plt.suptitle('Muestras agrupadas por clase', fontsize=14)
    plt.tight_layout()
    plt.show()

def standarization(X_train, X_test):
    """
    Aprende media y desvío sobre train, aplica la transformación a ambos.
    """
    media = np.mean(X_train, axis=0)   # media de cada pixel (shape: 784,)
    desvio = np.std(X_train, axis=0)   # desvío de cada pixel (shape: 784,)
    
    # Evitar división por cero en píxeles constantes (ej: fondo negro)
    desvio[desvio == 0] = 1
    
    X_train_std = (X_train - media) / desvio
    X_test_std  = (X_test  - media) / desvio  # usa la media/desvío del TRAIN
    
    return X_train_std, X_test_std, media, desvio



def stratified_split(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    
    X_train, X_test, y_train, y_test = [], [], [], []
    
    # Para cada clase, hacer el split por separado
    for clase in np.unique(y):
        idx = np.where(y == clase)[0]
        np.random.shuffle(idx)
        
        n_test = int(len(idx) * test_size)
        
        idx_test = idx[:n_test]
        idx_train = idx[n_test:]
        
        X_test.append(X[idx_test])
        X_train.append(X[idx_train])
        y_test.append(y[idx_test])
        y_train.append(y[idx_train])
    
    # Concatenar todas las clases
    X_train = np.concatenate(X_train)
    X_test  = np.concatenate(X_test)
    y_train = np.concatenate(y_train)
    y_test  = np.concatenate(y_test)
    
    # Mezclar para que no estén ordenados por clase
    train_idx = np.random.permutation(len(X_train))
    test_idx  = np.random.permutation(len(X_test))
    
    return X_train[train_idx], X_test[test_idx], y_train[train_idx], y_test[test_idx]
