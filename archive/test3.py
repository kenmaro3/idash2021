import pickle
import numpy as np


def calc_dist(x, centers):
  diff = centers - x
  diff = diff * diff
  dist = np.sum(diff, axis=1)
  print("============")
  print(dist)
  return dist

def calc_dists(xs, centers):
  res = []
  for x in xs:
    res.append(np.argmin(calc_dist(x, centers)))
  return res



if __name__ == "__main__":
  print("hello world")
  with open("kmean.pkl", "rb") as f:
    model = pickle.load(f)
  with open("knn.pkl", "rb") as f:
    model2 = pickle.load(f)

  print(model)
  centers = model.cluster_centers_
  print(centers.shape)


  with open("x_test.pkl", "rb") as f:
    x_test = pickle.load(f)
  
  with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

  print(x_test.shape)
  print(y_test.shape)

  
  #size = 10
  #res = calc_dists(x_test[:size], centers)
  #print(res)
  #print(y_test[:size])

  tmp1 = model.predict(x_test[:10]) 
  tmp2 = model2.predict(x_test[:10])
  print(tmp1)
  print(tmp2)

  
