from sklearn import datasets
iris = datasets.load_iris()
from sklearn.preprocessing import StandardScaler
standadizer = StandardScaler()
X = iris.data
y = iris.target
iris.target_names
xst = standadizer.fit_transform(X)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(xst, y)
new_observation = [[0.75,0.75,0.75,0.75],[1,1,1,1]]
knn.predict(new_observation)