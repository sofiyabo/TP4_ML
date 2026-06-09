import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_images(data, n_images=10, row_labels=None, titles=None, fixed_idx=None, n_cols=None):
    if isinstance(data, pd.DataFrame):
        pixel_cols = [c for c in data.columns if c.startswith("pixel_")]
        data = data[pixel_cols].values

    if n_cols is not None:
        n_images = n_cols

    single = not isinstance(data, list)
    if single:
        data = [data]

    n_rows = len(data)
    idx = fixed_idx if fixed_idx is not None else np.random.choice(len(data[0]), n_images, replace=False)

    has_labels = row_labels is not None and len(row_labels) > 0

    # Si hay etiquetas de fila, agregamos una columna extra a la izquierda para ellas
    total_cols = n_images + 1 if has_labels else n_images
    label_col_width = 1.2  # ancho de la columna de etiquetas
    img_col_width = 1.5

    fig_width = (label_col_width if has_labels else 0) + n_images * img_col_width
    fig_height = n_rows * 1.8

    # width_ratios: columna de etiqueta angosta + n_images columnas iguales
    if has_labels:
        width_ratios = [label_col_width] + [img_col_width] * n_images
    else:
        width_ratios = [img_col_width] * n_images

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=total_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": width_ratios} if has_labels else {},
        squeeze=False
    )

    for row_i, arr in enumerate(data):
        # Columna 0: etiqueta de fila
        if has_labels:
            ax_label = axes[row_i, 0]
            ax_label.axis("off")
            label = row_labels[row_i] if row_i < len(row_labels) else ""
            ax_label.text(
                0.95, 0.5, label,
                ha="right", va="center",
                fontsize=10, fontweight="bold",
                transform=ax_label.transAxes
            )

        # Columnas de imágenes
        for col_i in range(n_images):
            ax = axes[row_i, col_i + (1 if has_labels else 0)]
            ax.imshow(arr[idx[col_i]].reshape(28, 28), cmap="gray")
            ax.axis("off")

            # Título de clase encima, solo en primera fila
            if titles is not None and row_i == 0 and col_i < len(titles):
                ax.set_title(titles[col_i], fontsize=8, pad=3)

    plt.tight_layout(pad=0.5)
    plt.show()
    return idx

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
    
    # Evitar división por cero en píxeles constantes 
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
