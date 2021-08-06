import time
import sys
import numpy as np
import pandas as pd
import sklearn
import pickle

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score

def give_me_data(filename=None, pandas_object=None):

  if filename is not None and pandas_object is None:
      #with open('numerical_df.pkl', 'rb') as f:
      with open(filename, 'rb') as f:
        df = pickle.load(f)
  elif filename is None and pandas_object is not None:
      df = pandas_object


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

  return X,y

def load_df_for_test(size=500):
  filename = "Challenge/Challenge.fa"
  with open(filename, "r") as f:
    test = f.readlines()

  x = []
  y = []
  lens = []
  #for i in range(len(test)):
  for i in range(0, 0 + size*2):
    if i % 2 == 0:
      x.append(test[i])
    else:
      tmp = test[i].split("\n")[0]
      lens.append(len(tmp))
      y.append(tmp)

  for i in range(3999, 3999 + size*2):
    if i % 2 == 0:
      x.append(test[i])
    else:
      tmp = test[i].split("\n")[0]
      lens.append(len(tmp))
      y.append(tmp)

  for i in range(7999, 7999 + size*2):
    if i % 2 == 0:
      x.append(test[i])
    else:
      tmp = test[i].split("\n")[0]
      lens.append(len(tmp))
      y.append(tmp)

  for i in range(11999, 11999+size*2):
    if i % 2 == 0:
      x.append(test[i])
    else:
      tmp = test[i].split("\n")[0]
      lens.append(len(tmp))
      y.append(tmp)

  len_max = max(lens)

  print("here")
  print(len_max)
  for i in range(len(y)):
    diff = len_max - len(y[i])
    for j in range(diff):
      y[i] += "N"

  xs = [">B.1.526", ">B.1.1.7", ">B.1.427", ">P.1"]
  print(f'seq_len: {len(y[0])}')


  l = []
  c0, c1, c2, c3 = [], [], [], []
  df_lists = []

  for i in range(len(x)):
    df_list = []
    if x[i].startswith(xs[0]):
      df_list.append(0)
      tmp = x[i].split("_")[1].split("\n")[0]
      df_list.append(tmp)
      df_list.append(y[i])
    elif x[i].startswith(xs[1]):
      df_list.append(1)
      tmp = x[i].split("_")[1].split("\n")[0]
      df_list.append(tmp)
      df_list.append(y[i])
    elif x[i].startswith(xs[2]):
      df_list.append(2)
      tmp = x[i].split("_")[1].split("\n")[0]
      df_list.append(tmp)
      df_list.append(y[i])
    elif x[i].startswith(xs[3]):
      df_list.append(3)
      tmp = x[i].split("_")[1].split("\n")[0]
      df_list.append(tmp)
      df_list.append(y[i])
    else:
      raise

    df_lists.append(df_list)

  return df_lists

#def load_df(start, end):
def load_df():
  filename = "Challenge/Challenge.fa"
  with open(filename, "r") as f:
    test = f.readlines()

  x = []
  y = []
  lens = []
  for i in range(len(test)):
  #for i in range(start, end):
    if i % 2 == 0:
      x.append(test[i])
    else:
      tmp = test[i].split("\n")[0]
      lens.append(len(tmp))
      y.append(tmp)

  len_max = max(lens)

  print("here")
  print(len_max)
  for i in range(len(y)):
    diff = len_max - len(y[i])
    for j in range(diff):
      y[i] += "N"

  xs = [">B.1.526", ">B.1.1.7", ">B.1.427", ">P.1"]
  print(f'seq_len: {len(y[0])}')


  l = []
  c0, c1, c2, c3 = [], [], [], []
  df_lists = []

  for i in range(len(x)):
    df_list = []
    if x[i].startswith(xs[0]):
      df_list.append(0)
      tmp = x[i].split("_")[1].split("\n")[0]
      df_list.append(tmp)
      df_list.append(y[i])
    elif x[i].startswith(xs[1]):
      df_list.append(1)
      tmp = x[i].split("_")[1].split("\n")[0]
      df_list.append(tmp)
      df_list.append(y[i])
    elif x[i].startswith(xs[2]):
      df_list.append(2)
      tmp = x[i].split("_")[1].split("\n")[0]
      df_list.append(tmp)
      df_list.append(y[i])
    elif x[i].startswith(xs[3]):
      df_list.append(3)
      tmp = x[i].split("_")[1].split("\n")[0]
      df_list.append(tmp)
      df_list.append(y[i])
    else:
      raise

    df_lists.append(df_list)

  return df_lists



if __name__ == "__main__":
  df_list = load_df_for_test()
  columns = ["class", "id", "seq"]
  df = pd.DataFrame(data=df_list, columns=columns)
  df_s = df.sample(frac=1)
  df = df_s

  # each column in a dataframe is called a series
  classes = df.loc[:, 'class'].tolist()
  sequences = df.loc[:, 'seq'].tolist()

  dataset = {}

  skip = 50

  # Loop throught the sequences and split into individual nucleotides
  for i, seq in enumerate(sequences):
      # split into nucleotides, remove tab characters
      n_tmp = list(seq)

      nucleotides = []
      for j in range(0, len(n_tmp), skip):
          if n_tmp[j] != '\t':
              nucleotides.append(n_tmp[j])

      #nucleotides = [x for x in nucleotides if x != '\t']
      
      # Append class assignment
      nucleotides.append(classes[i])
      # add to dataset
      dataset[i] = nucleotides

  df = pd.DataFrame(dataset).T

  print(len(df))
  print(len(df.columns))

  #df.rename(columns={29906: 'Class'}, inplace=True)
  df.rename(columns={599: 'Class'}, inplace=True)

  with open('pp_data/categories.pkl', 'rb') as f:
    cats = pickle.load(f)

  print("start categorizes df...")
  t1 = time.time()
  for i in range(len(df.columns)-1):
    df[df.columns[i]] = pd.Categorical(df[df.columns[i]], categories=cats[i])


  print("start dummies df...")
  dummy_df = pd.get_dummies(df)
  print(dummy_df.head())
  t2 = time.time()
  print(f"dumming test took {t2-t1} sec")

  #with open('pp_data/onehot_test.pkl', 'wb') as f:
  #  pickle.dump(dummy_df, f)

  # load data
  #X, y = give_me_data("pp_data/onehot_test.pkl")
  X, y = give_me_data(pandas_object=dummy_df)

  # pca data
  data_file_name = f"pp_pca/pca_200.pkl"
  with open(data_file_name, "rb") as f:
    executor = pickle.load(f)

  X = executor.transform(X)
  X = X / np.sqrt(executor.explained_variance_)

  np.savetxt("pp_csv/x_test.csv", X, delimiter=',', fmt='%f')
  np.savetxt("pp_csv/y_test.csv", y, delimiter=',', fmt='%f')

  quit()


  # this is for debug python prediction 
  # clf data
  with open('lr_pkl/clf.pkl', 'rb') as f:
    clf = pickle.load(f)

  print("prediction start....")
  predictions = clf.predict(X)
  np.savetxt("/Users/kmihara/Documents/idash2021/seal/results/result_from_python.csv", predictions, delimiter=',', fmt='%f')
  print("prediction done....")
  print(accuracy_score(y, predictions))
  print(classification_report(y, predictions))

  

