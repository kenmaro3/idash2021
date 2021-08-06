import numpy as np
import pandas as pd
import pickle
import time

if __name__ == "__main__":
  from sklearn.decomposition import PCA
  from sklearn.decomposition import SparsePCA
  from sklearn.decomposition import NMF
  from sklearn.decomposition import TruncatedSVD
  from sklearn.manifold import TSNE

  comp_dim = 200
  is_pca = True

  if is_pca:
    executor = PCA(n_components=comp_dim)
    data_file_name = f"pp_pca/pca_{comp_dim}.pkl"
  else:
    raise


  with open('pp_data/onehot_train.pkl', 'rb') as f:
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

  with open(data_file_name, "wb") as f:
    pickle.dump(executor, f)

