import sys
import numpy as np
import pandas as pd
import sklearn
import pickle

import constants

def load_df(fasta_filename):
  filename = fasta_filename
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

  for i in range(len(y)):
    diff = len_max - len(y[i])
    for j in range(diff):
      y[i] += "N"

  xs = constants.xs
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
  df_list = load_df(constants.fasta_filename)
  columns = ["class", "id", "seq"]
  df = pd.DataFrame(data=df_list, columns=columns)
  df= df.sample(frac=1)

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
  print(len(df.columns))

  df.rename(columns={599: 'Class'}, inplace=True)

  #with open('pp_data/original_train.pkl', 'wb') as f:
  #  pickle.dump(df, f)

  print("deal with dummies and categories for train and test....")
  cats = []

  for i in range(len(df.columns)-1):
    tmp = set(df[df.columns[i]].unique().tolist())
    tmp = list(tmp)
    tmp.sort()
    cats.append(tmp)
    cat = tmp
    df[df.columns[i]] = pd.Categorical(df[df.columns[i]], categories=cat)


  dummy_df = pd.get_dummies(df)


  with open(constants.pp_data_category, 'wb') as f:
    pickle.dump(cats, f)
  with open(constants.pp_data_onehot_train, 'wb') as f:
    pickle.dump(dummy_df, f)
