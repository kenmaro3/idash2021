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
  is_pca = False
  is_spca = True
  is_nmf = False
  is_tsvd = False
  is_tsne = False

  if is_pca:
    comp_type = "pca"
  elif is_spca:
    comp_type = "spca"
  elif is_nmf:
    comp_type = "nmf"
  elif is_tsne:
    comp_type = "tsne"
  elif is_tsvd:
    comp_type = "tsvd"

  else:
    raise

  is_nc = False
  is_kn = True
  knn_size = 3
  compressed_dim = 2
  is_visualize = True
  if is_visualize:
    assert compressed_dim == 2

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

  if not is_tsne:
    data_file_name = f"{comp_type}_{compressed_dim}.pkl"
    with open(data_file_name, "rb") as f:
      executor = pickle.load(f)
    X = executor.transform(X)
  else:
    tsne_file_name = f"tsne_{compressed_dim}.pkl"
    with open(tsne_file_name, "rb") as f:
      X = pickle.load(f)

  if is_visualize:
    x0 = X[np.where(y==0)]
    x1 = X[np.where(y==1)]
    x2 = X[np.where(y==2)]
    x3 = X[np.where(y==3)]


    print("=============")
    print(x0.shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter(x0[:,0],x0[:,1], c='red')
    ax.scatter(x1[:,0],x1[:,1], c='blue')
    ax.scatter(x2[:,0],x2[:,1], c='green')
    ax.scatter(x3[:,0],x3[:,1], c='orange')

    ax.set_title('second scatter plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.savefig(f'figure_{comp_type}.png')


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



  ################################################
  ##### NearestCentroid #########################
  #clf = KMeans(n_clusters=4)
  if is_nc:
    model_name = f"nc_{comp_type}_{compressed_dim}.pkl"
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


  ################################################
  ##### KNeighborsClassifier #####################
  if is_kn:
    model_name = f"knn_{knn_size}_{comp_type}_{compressed_dim}.pkl"
    print("knn fit start....")
    clf = KNeighborsClassifier(n_neighbors=knn_size)
    clf.fit(X_train, y_train)
    print("knn fit done....")

    with open(model_name, "wb") as f:
      pickle.dump(clf, f)
    with open(model_name, "rb") as f:
      clf = pickle.load(f)

    print(clf.get_params())

    print("load done....")

    print("prediction start....")
    predictions = clf.predict(X_test)
    print("prediction done....")
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

