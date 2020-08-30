import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
print(X.shape, y.shape)
print(np.hstack([X, y[:, np.newaxis]]))

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plt.show()

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')


for slope_v, intercept_v in [
    (1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, slope_v * xfit + intercept_v, '-g')


plt.xlim(-1, 3.5)
plt.show()

clf = SVC(kernel='linear')
print(clf.fit(X, y))

print(np.hstack([X, y[:, np.newaxis], clf.predict(X)[:, np.newaxis]]))


print(metrics.accuracy_score(y, clf.predict(X)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.94, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

clf_svc = SVC(kernel='linear')
clf_svc.fit(X_train, y_train)

print(metrics.accuracy_score(y_test, clf_svc.predict(X_test)))

# Logistic Regression
clf_lr = LogisticRegression()
clf_lr.fit(X_train, y_train)
print(metrics.accuracy_score(y_test, clf_lr.predict(X_test)))


# kNN
knn_nn3 = KNeighborsClassifier(n_neighbors=3)
knn_nn3.fit(X_train, y_train)
print(metrics.accuracy_score(y_test, knn_nn3.predict(X_test)))
