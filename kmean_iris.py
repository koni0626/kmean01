from sklearn import cluster
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
data = iris['data']


model = cluster.KMeans(n_clusters=3)
model.fit(data)

labels = model.labels_

ldata = data[labels == 0]
plt.scatter(ldata[:, 2], ldata[:, 3],
            c='black', alpha=0.3, s=100, marker='o')

ldata = data[labels == 1]
plt.scatter(ldata[:, 2], ldata[:, 3],
            c='black', alpha=0.3, s=100, marker='^')

ldata = data[labels == 2]
plt.scatter(ldata[:, 2], ldata[:, 3],
            c='black', alpha=0.3, s=100, marker='*')
            
plt.xlabel(iris['feature_names'][2])
plt.xlabel(iris['feature_names'][3])

plt.show()

def test_func():
    test = np.zeros(4)
    test[0] = 1
    test[1] = 0
    test[2] = 1
    test[3] = 0

    data = np.zeros(4)
    data[0] = 10
    data[1] = 20
    data[2] = 30
    data[3] = 40

    print(data[test == 0])
