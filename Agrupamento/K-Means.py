#%%
from sklearn.cluster import KMeans  
from sklearn.datasets import load_iris

#%%
df = load_iris()
atributos = df.data
df

#%%
# definindo a quantidade de clusters

# 1- Pelo target que já sabemos
qtd_clusters = len(df.target_names)

# 2- Método Elbow
import matplotlib.pyplot as plt
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans.fit(atributos)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS')
plt.show()

#%%
kmeans = KMeans(n_clusters=qtd_clusters)
kmeans.fit(atributos)
kmeans.labels_

#%%
data = [
    [4.12, 3.4, 1.6, 0.7],
    [5.2, 5.8, 5.2, 6.7],
    [3.1, 3.5, 3.3, 3.0]
]
kmeans.predict(data)

#%%
plt.scatter(atributos[:, 0], atributos[:, 1], s=50, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.title('Agrupamento de íris')
plt.xlabel('Comprimento da sépala (cm)')
plt.ylabel('Largura da sépala (cm)')

plt.show()

#%%
plt.scatter(atributos[:, 2], atributos[:, 3], s=50, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s=300, c='red', label='Centroids')
plt.title('Agrupamento de íris')
plt.xlabel('Comprimento da pétala (cm)')
plt.ylabel('Largura da pétala (cm)')

plt.show()
