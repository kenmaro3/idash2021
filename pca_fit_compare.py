import pickle
import numpy as np
import sys

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

from data import give_me_data, give_me_visual

if __name__ == "__main__":
  # load data
  X, y = give_me_data()

  # load trained pca class
  data_file_name = f"pca_200.pkl"
  with open(data_file_name, "rb") as f:
    executor = pickle.load(f)

  ######################################
  ## use executor property to do the same with fit
  # tmp buffer
  tmp_X = X
  tmp0 = tmp_X[0] - executor.mean_
  print("whiten")
  print(executor.whiten)
  tmp0 = np.dot(executor.components_, tmp0)

  ## actualy this operation is like normalization
  #if executor.whiten:
  tmp0 = tmp0 / np.sqrt(executor.explained_variance_)
  print(tmp0)

  ######################################
  ## compare the result with usual fit
  X = executor.transform(X[:10])
  print("here")
  print(X[0])
  sys.exit()
