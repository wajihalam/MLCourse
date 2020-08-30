from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression

irisDS = load_iris()

# save the feature matrix in X
x = irisDS.data

# save the ground truth vector in y
y = irisDS.target

print(x.shape, y.shape)


# create a default classifier and print the default parameters
knn_nn1 = KNeighborsClassifier(n_neighbors=1)
print(knn_nn1.fit(x, y))

# let's test it on the first observation
print(knn_nn1.predict([[5.1, 3.5, 1.4, 0.2]]))

# also predict on multiple vectors
x1 = np.array([5, 3.7, 1.4, 0])
x2 = np.array([7, 3, 4, 1])
print(knn_nn1.predict([x1, x2]))

# try predicting with different parameter
knn_nn5 = KNeighborsClassifier(n_neighbors=5)
knn_nn5.fit(x, y)
print(knn_nn5.predict([x[1, :]]))
print(knn_nn5.predict([x1, x2]))


# Linear regression
LR_estimator = LogisticRegression()
LR_estimator.fit(x, y)
LR_estimator.predict([x[1, :], x1, x2])

