import time
import sys
import numpy as np
import pandas as pd
import pickle

import constants

def give_me_data(filename=None, pandas_object=None):

  if filename is not None and pandas_object is None:
      with open(filename, 'rb') as f:
        df = pickle.load(f)
  elif filename is None and pandas_object is not None:
      df = pandas_object

  X = df.to_numpy()

  return X


def load_df(fasta_filename):
  with open(fasta_filename, "r") as f:
    test = f.readlines()

  x = []
  y = []
  lens = []
  for i in range(len(test)):
    if i % 2 == 0:
      x.append(test[i])
    else:
      tmp = test[i].split("\n")[0]
      lens.append(len(tmp))
      y.append(tmp)
  len_max = max(lens)

  for i in range(len(y)):
    if(len(y[i]) < 29906):
      diff = 29906- len(y[i])
      for j in range(diff):
        y[i] += "N"
    else:
      y[i] = y[i][:29906]

  xs = [">tag"]

  df_lists = []

  for i in range(len(x)):
    df_list = []
    if x[i].startswith(xs[0]):
      tmp = x[i].split("_")[1].split("\n")[0]
      df_list.append(tmp)
      df_list.append(y[i])
    else:
      raise

    df_lists.append(df_list)

  return df_lists


def get_onehot_df(df):
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
      dataset[i] = nucleotides

  df = pd.DataFrame(dataset).T


  with open(constants.pp_data_category, 'rb') as f:
    cats = pickle.load(f)

  print("start categorizes df...")
  for i in range(len(df.columns)):
    df[df.columns[i]] = pd.Categorical(df[df.columns[i]], categories=cats[i])

  print("start dummies df...")
  dummy_df = pd.get_dummies(df)

  return dummy_df


if __name__ == "__main__":
  print("\n\n==================================================================")
  from pyfiglet import Figlet
  f = Figlet(font="slant")
  msg = f.renderText("HELLO, IDASH")
  print(msg)
  print("\n==================================================================")
  f = Figlet(font='small')
  msg = f.renderText("TEAM EAGLYS")
  print(msg)
  f = Figlet(font='small')
  msg = f.renderText("(KENMARO")
  print(msg)
  msg = f.renderText("-RYOHEI)")
  print(msg)
  print("\n==================================================================")
  print("checking input fa file...")
  #filename = "Challenge/Challenge.fa"
  assert(len(sys.argv)==2)
  filename = sys.argv[1]
  #assert(filename.endswith("fa"))

  print("==================================================================")
  print("loading to df...")
  df_list = load_df(filename)
  columns = ["id", "seq"]
  df = pd.DataFrame(data=df_list, columns=columns)
  

  print("==================================================================")
  print("get one hot encoding...")
  dummy_df = get_onehot_df(df)
  t2 = time.time()

  print("==================================================================")
  print("make it into X, y form...")
  X = give_me_data(pandas_object=dummy_df)

  print("==================================================================")
  print("apply pca...")
  with open(constants.pp_pca_model, "rb") as f:
    executor = pickle.load(f)

  X = executor.transform(X)
  X = X / np.sqrt(executor.explained_variance_)

  print("==================================================================")
  print("write pcaed data to csv for coming c++...")
  np.savetxt(constants.preprocessed_test_x, X, delimiter=',', fmt='%f')