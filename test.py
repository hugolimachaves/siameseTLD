X = [[0,1], [1,10], [2,9], [3,8]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y) 

print(neigh.predict([[-10,10]]))

print(neigh.predict_proba([[-10,10]]))



