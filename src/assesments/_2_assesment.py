import numpy as np
from matplotlib import pyplot as plt  # библиотека для визуализации
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run():
    # Загружаем датасет
    mnist = fetch_openml('mnist_784', version=1)
    x = mnist.data

    # Проверить на выбросы, пропуски.
    print("Пропуски в данных:", np.isnan(x).sum())

    # Стандартизация данных
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Вычислить Silhouette Score и Inertia.
    # Выбрать K, при котором Silhouette Score максимален и Inertia оптимальна. (Подумайте как сделать процесс выбора числа кластеров полностью автоматическим)
    # !!!!!!!!! Очень долгий процесс, раскоменчивать для запуска
    best_cluster_dict = get_best_model(x_scaled)
    print(f"Лучшее значение K: {best_cluster_dict['best_k']}")

    # Определить, сколько главных компонент объясняют 95% дисперсии данных и перевести данные в эту размерность.
    # Применение метода главных компонент
    pca = PCA()
    pca.fit(x_scaled)

    # Определение числа компонент, объясняющих 95% дисперсии
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= 0.95) + 1
    print(f"Число компонент, объясняющих 95% дисперсии: {n_components}")

    # Перевод данных в новое пространство
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x_scaled)

    # Повторить кластеризацию K-Means с использованием преобразованных данных.
    pca_cluster_dict = get_cluster(x_pca, best_cluster_dict['best_k'])

    # Отрисовать данные через 2 главные компоненты.
    draw_cluster(x_pca, pca_cluster_dict['kmeans'], 'MNIST после PCA')
    draw_cluster(x_scaled, best_cluster_dict['best_model'], 'MNIST до PCA')


def get_cluster(x_scaled, k: int) -> dict[str, any]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)
    silhouette_avg = silhouette_score(x_scaled, kmeans.labels_)
    inertia = kmeans.inertia_

    print(f"K = {k}, Silhouette Score = {silhouette_avg}, Inertia = {inertia}")

    return {'kmeans': kmeans, 'silhouette_avg': silhouette_avg, 'inertia': inertia}


def get_best_model(x_scaled) -> dict[str, any]:
    result_dict = {'best_score': -1}

    for k in range(2, 11):
        cluster_dict = get_cluster(x_scaled, k)

        if cluster_dict['silhouette_avg'] > result_dict['best_score']:
            result_dict['best_score'] = cluster_dict['silhouette_avg']
            result_dict['best_k'] = k
            result_dict['best_model'] = cluster_dict['kmeans']

    return result_dict


def draw_cluster(x, cluster, title: str):
    plt.figure(figsize=(10, 6))
    plt.scatter(x[:, 0], x[:, 1], c=cluster.labels_, cmap='viridis', s=10)
    plt.title(title)
    plt.colorbar(label='Cluster')
    plt.show()
