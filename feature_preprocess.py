import numpy as np
import pandas as pd
import pickle

if __name__ == "__main__":
  from sklearn.decomposition import PCA
  from sklearn.decomposition import SparsePCA
  from sklearn.decomposition import NMF
  from sklearn.decomposition import TruncatedSVD
  from sklearn.manifold import TSNE

  comp_dim = 2
  is_pca = False
  is_spca = True
  is_nmf = False
  is_tsvd = False
  is_tsne = False

  if is_pca:
    executor = PCA(n_components=comp_dim)
    data_file_name = f"pca_{comp_dim}.pkl"
  elif is_spca:
    executor = SparsePCA(n_components=comp_dim)
    data_file_name = f"spca_{comp_dim}.pkl"
  elif is_nmf:
    executor = NMF(n_components=comp_dim)
    data_file_name = f"nmf_{comp_dim}.pkl"
  elif is_tsvd:
    executor = TruncatedSVD(n_components=comp_dim)
    data_file_name = f"tsvd_{comp_dim}.pkl"
  elif is_tsne:
    executor = TSNE(n_components=comp_dim)
    data_file_name = f"tsne_{comp_dim}.pkl"



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
    executor.fit(X)

    with open(data_file_name, "wb") as f:
      pickle.dump(executor, f)

  if is_tsne:
    x = executor.fit_transform(X)
    print("here")
    print(x.shape)
    with open(data_file_name, "wb") as f:
      pickle.dump(x, f)

