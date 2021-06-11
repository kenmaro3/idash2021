import pickle
import numpy as np
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score

import matplotlib.pyplot as plt


if __name__ == "__main__":
  with open('numerical_df.pkl', 'rb') as f:
    df = pickle.load(f)

  print(len(df.columns))


  c = []
  for i in range(len(df)):
    if df['Class_0'][i] == 1:
      c.append(0)
    elif df['Class_1'][i] == 1:
      c.append(1)
    elif df['Class_2'][i] == 1:
      c.append(2)
    elif df['Class_3'][i] == 1:
      c.append(3)
    else:
      raise

  df = df.drop(columns=['Class_0'])
  df = df.drop(columns=['Class_1'])
  df = df.drop(columns=['Class_2'])
  df = df.drop(columns=['Class_3'])
  df["Class"] = c
  print(df.head())


  X = df.drop(['Class'], axis=1).to_numpy()
  y = df['Class'].to_numpy()

  #with open("pca_2.pkl", "rb") as f:
  #  pca = pickle.load(f)
  #X = pca.transform(X)

  with open("tsne_2_x.pkl", "rb") as f:
    X = pickle.load(f)


  #x0 = X[np.where(y==0)]
  #x1 = X[np.where(y==1)]
  #x2 = X[np.where(y==2)]
  #x3 = X[np.where(y==3)]


  #print("=============")
  #print(x0.shape)
  #print(x1.shape)
  #print(x2.shape)
  #print(x3.shape)


  #fig = plt.figure()
  #ax = fig.add_subplot(1,1,1)

  #ax.scatter(x0[:,0],x0[:,1], c='red')
  #ax.scatter(x1[:,0],x1[:,1], c='blue')
  #ax.scatter(x2[:,0],x2[:,1], c='green')
  #ax.scatter(x3[:,0],x3[:,1], c='orange')

  #ax.set_title('second scatter plot')
  #ax.set_xlabel('x')
  #ax.set_ylabel('y')

  #plt.show()
  #plt.savefig('figure.png')

  #sys.exit()

  # Split the data into training and test dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
  with open("x_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
  with open("x_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
  with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
  with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)



  #clf = KMeans(n_clusters=4)
  clf = NearestCentroid()

  print("kmean fit start....")
  clf.fit(X_train, y_train)
  print("kmean fit done....")
  #with open("nc1000.pkl", "wb") as f:
  #  pickle.dump(clf, f)

  print("prediction start....")
  predictions = clf.predict(X_test)
  print("prediction done....")
  print(accuracy_score(y_test, predictions))
  print(classification_report(y_test, predictions))
  print("======")
  print(clf.centroids_)
  print(clf.classes_)
  print(clf.centroids_.shape)


  print("knn fit start....")
  clf = KNeighborsClassifier(n_neighbors=30)
  clf.fit(X_train, y_train)
  print("knn fit done....")

  #with open("knn1000.pkl", "wb") as f:
  #  pickle.dump(clf, f)
  #with open("kmean.pkl", "rb") as f:
  #  clf = pickle.load(f)

  #with open("knn.pkl", "rb") as f:
  #  clf = pickle.load(f)
  print(clf.get_params())

  print("load done....")

  print("prediction start....")
  predictions = clf.predict(X_test)
  print("prediction done....")
  print(accuracy_score(y_test, predictions))
  print(classification_report(y_test, predictions))

  ## Test the algorithm on the validation dataset
  #for name, model in models:
  #    model.fit(X_train, y_train)
  #    predictions = model.predict(X_test)
  #    print(name)
  #    print(accuracy_score(y_test, predictions))
  #    print(classification_report(y_test, predictions))
