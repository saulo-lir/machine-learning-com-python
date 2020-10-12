import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs # Método que contém dados de testes

x, y = make_blobs(n_samples = 200, centers = 4) # x = Registros, y = Clesters

plt.scatter(x[:,0], x[:,1])
plt.show()

kmeans = KMeans(n_clusters = 4)
kmeans.fit(x)

previsoes = kmeans.predict(x)
plt.scatter(x[:,0], x[:,1], c = previsoes)
plt.show()