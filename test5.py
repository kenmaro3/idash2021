import pickle
import numpy as np
import sys
from sklearn.model_selection import KFold, train_test_split, cross_val_score
import scipy.stats as stats


def apply_pca():
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

  #with open("pca_2.pkl", "rb") as f:
  #  pca = pickle.load(f)
  #  X = pca.transform(X)
  #print("pca aplied..")
  #print(X.shape)
  with open("tsne_2_x.pkl", "rb") as f:
    X = pickle.load(f)
  print("tsne aplied..")
  # Split the data into training and test dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
  with open("x_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
  with open("x_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
  with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
  with open("y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)



if __name__ == "__main__":
  apply_pca()
  
  with open("x_train.pkl", "rb") as f:
    X_train = pickle.load(f)
  with open("x_test.pkl", "rb") as f:
    X_test = pickle.load(f)
  with open("y_train.pkl", "rb") as f:
    y_train = pickle.load(f)
  with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)



  n = 10
  ns = [3]
  for n in ns:
    right = 0
    tot = 0
    cannot = []
    for i in range(len(X_test)):
      test = X_test[i]

      tmp1 = X_train - test
      tmp2 = tmp1 * tmp1
      tmp3 = np.sum(tmp2, axis=1)
      tmp4 = tmp3.reshape(-1) 
      idx = np.argpartition(tmp4, n)
      res = y_train[idx[:n]]
      mode_val, mode_num = stats.mode(res)

      if mode_val[0] == y_test[i]:
        right += 1
      else:
        cannot.append(i)


      tot += 1

    print(right)
    #print(tot)
    print(f'\nn: {n}')

    print(right/tot)

    print("====")
    print(cannot)
    #print(y_test[cannot])
    #print(X_test[cannot])



