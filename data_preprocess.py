import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
import pickle

def load_df_for_test(size=100):
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

  for i in range(1999, 1999 + size*2):
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

  for i in range(5999, 5999+size*2):
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
  df_list = load_df()
  columns = ["class", "id", "seq"]
  df = pd.DataFrame(data=df_list, columns=columns)
  df_s = df.sample(frac=1)
  df = df_s.iloc[:6000, :]
  df2 = df_s.iloc[6000:, :]

  # each column in a dataframe is called a series
  classes = df.loc[:, 'class'].tolist()
  classes2 = df2.loc[:, 'class'].tolist()

  sequences = df.loc[:, 'seq'].tolist()
  sequences2 = df2.loc[:, 'seq'].tolist()


  dataset = {}
  dataset2 = {}

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

  for i, seq in enumerate(sequences2):
      # split into nucleotides, remove tab characters
      n_tmp = list(seq)

      nucleotides = []
      for j in range(0, len(n_tmp), skip):
          if n_tmp[j] != '\t':
              nucleotides.append(n_tmp[j])

      #nucleotides = [x for x in nucleotides if x != '\t']
      
      # Append class assignment
      nucleotides.append(classes2[i])
      
      # add to dataset
      dataset2[i] = nucleotides

  #for i, seq in enumerate(sequences2):
  #    # split into nucleotides, remove tab characters
  #    nucleotides = list(seq)
  #    nucleotides = [x for x in nucleotides if x != '\t']
  #    
  #    # Append class assignment
  #    nucleotides.append(classes[i])
  #    
  #    # add to dataset
  #    dataset2[i] = nucleotides
      
  df = pd.DataFrame(dataset).T  
  df2 = pd.DataFrame(dataset2).T  
  print("here")
  print(len(df.columns))

  #df.rename(columns={29906: 'Class'}, inplace=True)
  df.rename(columns={599: 'Class'}, inplace=True)
  df2.rename(columns={599: 'Class'}, inplace=True)
  assert(len(df.columns) == len(df2.columns))
  print(df.columns)

  with open('pp_data/original_train.pkl', 'wb') as f:
    pickle.dump(df, f)

  print("deal with dummies and categories for train and test....")
  cats = []

  for i in range(len(df.columns)-1):
    #cat = set(df[df.columns[i]].unique().tolist())
    tmp = set(df[df.columns[i]].unique().tolist())
    tmp = list(tmp)
    tmp.sort()
    cats.append(tmp)
    cat = tmp
    #cat = set(tmp)
    df[df.columns[i]] = pd.Categorical(df[df.columns[i]], categories=cat)
    df2[df2.columns[i]] = pd.Categorical(df2[df2.columns[i]], categories=cat)


  dummy_df = pd.get_dummies(df)
  dummy_df2 = pd.get_dummies(df2)


  with open('pp_data/categories.pkl', 'wb') as f:
    pickle.dump(cats, f)

  with open('pp_data/onehot_train.pkl', 'wb') as f:
    pickle.dump(dummy_df, f)
  with open('pp_data/original_test.pkl', 'wb') as f:
    pickle.dump(df2, f)
  with open('pp_data/onehot_test.pkl', 'wb') as f:
    pickle.dump(dummy_df2, f)
