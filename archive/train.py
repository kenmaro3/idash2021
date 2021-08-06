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

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from data import give_me_data, give_me_visual


if __name__ == "__main__":
  is_pca = True
  is_spca = False
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
  is_kn = False
  is_lr = True
  knn_size = 3
  compressed_dim = 2
  is_visualize = True
  if is_visualize:
    assert compressed_dim == 2

  X, y = give_me_data()

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
    give_me_visual(X, y, f'figure_{comp_type}.png')


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


  if is_lr:
    model_name = f"nc_{comp_type}_{compressed_dim}.pkl"
    clf = LogisticRegression(random_state=0)

    #print("kmean fit start....")
    #clf.fit(X_train, y_train)
    #print("kmean fit done....")


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

