import pickle
import numpy as np
import sys

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score

from data import give_me_data, give_me_visual

if __name__ == "__main__":
  ## load data
  X, y = give_me_data("pp_data/onehot_train.pkl")
  ## load trained pca class
  data_file_name = f"pp_pca/pca_200.pkl"
  with open(data_file_name, "rb") as f:
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

  with open('lr_pkl/clf.pkl', 'wb') as f:
    pickle.dump(clf, f)

  print("here")
  print(clf.intercept_)

  np.save("lr_npy/weight.npy", clf.coef_, allow_pickle=False)
  np.save("lr_npy/bias.npy", clf.intercept_, allow_pickle=False)

  np.savetxt("lr_csv/weight.csv", clf.coef_, delimiter=',', fmt='%f')
  np.savetxt("lr_csv/bias.csv", clf.intercept_, delimiter=',', fmt='%f')

  ###################################################################
  #executor.whiten = True
  #X = executor.transform(tmp_X)
  #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

  #clf = LogisticRegression(random_state=0).fit(X_train, y_train)
  #score = clf.score(X_test, y_test)
  #print("with normalization======================================")
  #print("done")
  #print(score)
