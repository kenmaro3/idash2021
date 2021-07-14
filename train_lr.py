import pickle
import numpy as np
import sys

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

from data import give_me_data, give_me_visual

if __name__ == "__main__":
  X, y = give_me_data()
  data_file_name = f"pca_200.pkl"
  with open(data_file_name, "rb") as f:
    executor = pickle.load(f)
  X = executor.transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

  clf = LogisticRegression(random_state=0).fit(X_train, y_train)
  score = clf.score(X_test, y_test)
  print("done")
  print(score)
