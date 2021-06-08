import pickle
import numpy as np
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score


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

  # Split the data into training and test dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)


  scoring = 'accuracy'

  # Define models to train
  #names = ['K Nearest Neighbors', 'Gaussian Process', 'Decision Tree', 'Random Forest',
  #         'Neural Network', 'AdaBoost', 'Naive Bayes', 'SVM Linear', 'SVM RBF', 'SVM Sigmoid']
  names = ['K Nearest Neighbors', 'Decision Tree', 'Random Forest',
           'Neural Network', 'AdaBoost', 'Naive Bayes', 'SVM Linear', 'SVM RBF', 'SVM Sigmoid']

  classifiers = [
      #KNeighborsClassifier(n_neighbors=3),
      #GaussianProcessClassifier(1.0 * RBF(1.0)),
      #DecisionTreeClassifier(max_depth=5),
      #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
      MLPClassifier(alpha=1, max_iter=500),
      AdaBoostClassifier(),
      #GaussianNB(),
      SVC(kernel='linear'),
      #SVC(kernel='rbf'),
      #SVC(kernel='sigmoid')
  ]

  models = zip(names, classifiers)

  # Evaluate each model in turn
  results = []
  names = []

  for name, model in models:
      kfold = KFold(n_splits=10, shuffle=True)
      cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
      results.append(cv_results)
      names.append(name)
      msg = '{0}:  {1}  ({2})'.format(name, cv_results.mean(), cv_results.std())
      print(msg)

  models = zip(names, classifiers)
  with open("models.pkl", "wb") as f:
    pickle.dump(models, f)

  # Test the algorithm on the validation dataset
  for name, model in models:
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      print(name)
      print(accuracy_score(y_test, predictions))
      print(classification_report(y_test, predictions))
