#/usr/bin/python3
# importing a dataset
from sklearn import datasets
iris = datasets.load_iris()

# taking features
X = iris.data
# taking lables
y = iris.target

from sklearn.cross_validation import train_test_split
# Partitioning into train data as 90% and test data as 10%
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = .1)

from sklearn.cluster import KMeans

# setting number of clusters to 3, since the dataset contains 3 different types of flowers
kmeans_cluster = KMeans(n_clusters=3)

kmeans_cluster.fit(X_train)
# We can print and see the cluster labels vs actual lables
# print("Cluster Labels ", kmeans_cluster.labels_)
# print("Actual Labels ", y_train)
# We can check how accuracy is with an utility
from sklearn.metrics import accuracy_score

print("Training Accuracy ", accuracy_score(y_train, kmeans_cluster.labels_))

predictions = kmeans_cluster.predict(X_test)
# print("Predictions ", predictions)

# Accuracy test with actual to predicted values
print("Testing Accurarcy ", accuracy_score(y_test, predictions))