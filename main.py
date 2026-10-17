# Author: Márcio Belloni

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from elbow import elbow

dados = sns.load_dataset('mpg')
df = pd.DataFrame(dados)

df_subconjunto = df.sample(30, random_state=42)

colunas_numericas = ['mpg', 'horsepower', 'weight']
dados_para_aglomeracao = df_subconjunto[colunas_numericas]

dados_para_aglomeracao.fillna(dados_para_aglomeracao.mean(), inplace=True)

elbow(dados_para_aglomeracao)

dados_para_aglomeracao_sem_nan = dados_para_aglomeracao.dropna()

num_clusters = 4
limite_distancia = 1000

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_subconjunto['cluster'] = kmeans.fit_predict(dados_para_aglomeracao_sem_nan)

matriz_ligacao = linkage(dados_para_aglomeracao_sem_nan, method='ward')

plt.figure(figsize=(15, 5))
dendrogram(matriz_ligacao, labels=df_subconjunto['name'].tolist(), leaf_rotation=90, leaf_font_size=8, color_threshold=limite_distancia)
plt.title('Dendrograma de Aglomeração Hierárquica com Clusters K-Means')
plt.xlabel('Marca')
plt.ylabel('Rendimento')
plt.show()

print("Clusters:")
print(df_subconjunto[['name', 'cluster']])