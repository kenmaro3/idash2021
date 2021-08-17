import pickle
import numpy as np
import sys

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score

import constants

def give_me_data(filename=None, pandas_object=None):

  if filename is not None and pandas_object is None:
      #with open('numerical_df.pkl', 'rb') as f:
      with open(filename, 'rb') as f:
        df = pickle.load(f)
  elif filename is None and pandas_object is not None:
      df = pandas_object



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

  return X,y

if __name__ == "__main__":
  ## load data
  X, y = give_me_data(filename=constants.pp_data_onehot_train)
  ## load trained pca class
  with open(constants.pp_pca_model , "rb") as f:
    executor = pickle.load(f)

  ##################################################################
  executor.whiten = False
  X = executor.transform(X)
  X = X / np.sqrt(executor.explained_variance_)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

  clf = LogisticRegression(random_state=0).fit(X_train, y_train)
  score = clf.score(X_test, y_test)
  print("without normalization======================================")
  print("done")
  print(score)
  predictions = clf.predict(X_test)
  print(accuracy_score(y_test, predictions))
  print(classification_report(y_test, predictions))

  with open(constants.pp_lr_model, 'wb') as f:
    pickle.dump(clf, f)

  #print("this is trained lr parameters...")
  #print(clf.coef_)
  #print(clf.intercept_)

  np.savetxt(constants.pp_lr_model_weight, clf.coef_, delimiter=',', fmt='%f')
  np.savetxt(constants.pp_lr_model_bias, clf.intercept_, delimiter=',', fmt='%f')

  ###################################################################
  #executor.whiten = True
  #X = executor.transform(tmp_X)
  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

  #clf = LogisticRegression(random_state=0).fit(X_train, y_train)
  #score = clf.score(X_test, y_test)
  #print("with normalization======================================")
  #print("done")
  #print(score)
