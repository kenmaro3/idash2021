from test import load_data, ttsplit
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import pickle

def load_df():
  filename = "Challenge/Challenge.fa"
  with open(filename, "r") as f:
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
  df_list = load_df()
  print(len(df_list))
  columns = ["class", "id", "seq"]
  df = pd.DataFrame(data=df_list, columns=columns)
  print(df.head())


  # each column in a dataframe is called a series
  classes = df.loc[:, 'class']
  classes.value_counts()


  sequences = df.loc[:, 'seq'].tolist()

  dataset = {}

  # Loop throught the sequences and split into individual nucleotides
  for i, seq in enumerate(sequences):
      # split into nucleotides, remove tab characters
      nucleotides = list(seq)
      nucleotides = [x for x in nucleotides if x != '\t']
      
      # Append class assignment
      nucleotides.append(classes[i])
      
      # add to dataset
      dataset[i] = nucleotides
      
  df = pd.DataFrame(dataset).T  
  print(df.head())

  df.rename(columns={29906: 'Class'}, inplace=True)

  df.describe()

  with open('df.pkl', 'wb') as f:
    pickle.dump(df, f)

  #sys.exit()

  with open('df.pkl', 'rb') as f:
    df = pickle.load(f)

  print(df.head())


  series = []

  for name in df.columns:
      series.append(df[name].value_counts())

  info = pd.DataFrame(series)
  details = info.T
  print(details)

  numerical_df = pd.get_dummies(df)
  print(numerical_df.head())

  print(len(numerical_df))

  
  with open('numerical_df.pkl', 'wb') as f:
    pickle.dump(numerical_df, f)

  

