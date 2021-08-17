import numpy as np
import pandas as pd
import pickle
import time

import constants

if __name__ == "__main__":
  from sklearn.decomposition import PCA

  comp_dim = 200
  is_pca = True

  if is_pca:
    executor = PCA(n_components=comp_dim)
  else:
    raise


  with open(constants.pp_data_onehot_train, 'rb') as f:
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


  t1 = time.time()

  executor.fit(X)
  t2 = time.time()
  print(f"fitting lr took {t2-t1} sec")

  with open(constants.pp_pca_model, "wb") as f:
    pickle.dump(executor, f)

