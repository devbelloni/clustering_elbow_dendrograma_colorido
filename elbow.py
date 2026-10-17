# Author: Márcio Belloni

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def elbow(dados_padronizado):
    inercia = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(dados_padronizado)
        inercia.append(kmeans.inertia_)

    plt.plot(range(1, 11), inercia, marker='o')
    plt.title('Método de Elbow')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.show()