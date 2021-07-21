import pickle
import numpy as np
import sys

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

from data import give_me_data, give_me_visual

if __name__ == "__main__":
  # load trained pca class
  data_file_name = f"pca_200.pkl"
  with open(data_file_name, "rb") as f:
    executor = pickle.load(f)

  np.savetxt('pca_csv/pca_200_mean.csv', executor.mean_)
  np.savetxt('pca_csv/pca_200_components.csv', executor.components_)
  np.savetxt('pca_csv/pca_200_variance.csv', executor.explained_variance_)

  sys.exit()
