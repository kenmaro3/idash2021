import numpy as np
import pandas as pd
import pickle

if __name__ == "__main__":
  from sklearn.decomposition import PCA
  from sklearn.manifold import TSNE

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

  #with open("pca_x.pkl", "rb") as f:
  #  pcax = pickle.load(f)
  #print(pcax.shape)



  #pca_executor = PCA(n_components=2)
  #pca_executor.fit(X)
  #with open("pca_2.pkl", "wb") as f:
  #  pickle.dump(pca_executor, f)
  from sklearn.manifold import TSNE
  clf = TSNE(n_components=2)
  x = clf.fit_transform(X)
  print("here")
  print(x.shape)
  with open("tsne_2_x.pkl", "wb") as f:
    pickle.dump(x, f)

  #tsne = TSNE(n_components=500, random_state=41)
  #X_reduced = tsne.fit_transform(X)

  #with open("tsne_x.pkl", "wb") as f:
  #  pickle.dump(X_reduced, f)
