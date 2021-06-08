import hashlib
import time
import numpy as np

def load_data():
  filename = "Challenge/Challenge.fa"
  with open(filename, "r") as f:
    test = f.readlines()

  x = []
  y = []
  for i in range(len(test)):
    if i % 2 == 0:
      x.append(test[i])
    else:
      y.append(test[i])

  xs = [">B.1.526", ">B.1.1.7", ">B.1.427", ">P.1"]
  print(f'seq_len: {len(y[0])}')


  l = []
  c0, c1, c2, c3 = [], [], [], []

  for i in range(len(x)):
    if x[i].startswith(xs[0]):
      c0.append(y[i])
      l.append(0)
    elif x[i].startswith(xs[1]):
      c1.append(y[i])
      l.append(1)
    elif x[i].startswith(xs[2]):
      c2.append(y[i])
      l.append(2)
    elif x[i].startswith(xs[3]):
      c3.append(y[i])
      l.append(3)
    else:
      raise

  return c0, c1, c2, c3, l
  #return ttsplit(c0), ttsplit(c1), ttsplit(c2), ttsplit(c3), l

def pp(k, c0, bl, hash_func):
  test1 = []
  for i in range(len(c0)):
    test1.append(ppone(k, c0[i], bl, hash_func))

  return test1


def ppone(k, c0, bl, hash_func):
  tmp = c0
  test2 = []
  for j in range(0, len(tmp), k):
    tmp1 = tmp[j:j+k]
    #tmp2 = int.from_bytes(hash_func(tmp1.encode()).digest()[:bl ], 'little')
    tmp2 = hash_func(tmp1.encode()).hexdigest()[:bl ]
    test2.append(tmp2)
  return sorted(test2)



def queryone(x, c):
  same = 0
  tot = 0
  for i in range(len(c)):
    if x[0] == c[i][0]:
      same += 1
    tot += 1

  return same/tot


def query(xs, c):
  return [queryone(xs[i], c) for i in range(len(xs))]



def ttsplit(x, num=1600):
  return x[:num], x[num:]

def test(xs, c0t, c1t, c2t, c3t):
  tmp = []
  for i in range(len(xs)):
    print(f'testing {i}')
    tmp1 = testone(xs[i], c0t, c1t, c2t, c3t)
    print(tmp1)
    tmp.append(tmp1)

  #tmp = [testone(xs[i], c0t, c1t, c2t, c3t) for i in range(len(xs))]
  return np.argmax(tmp, axis=1)

def testone(x, c0t, c1t, c2t, c3t):
  res = []
  res.append(queryone(x, c0t))
  res.append(queryone(x, c1t))
  res.append(queryone(x, c2t))
  res.append(queryone(x, c3t))
  return res

def test_all(c0t, c1t, c2t, c3t):
  res = []
  test_source = [c0_test, c1_test, c2_test, c3_test]
  for x in test_source:
    x = pp(k, x, bl, hashlib.blake2s)

    t1 = time.time()
    resx = test(x, c0t, c1t, c2t, c3t)
    t2 = time.time()
    print(f'test time: {t2-t1}')
    ans = 0
    acc = 0
    tot = 0
    for i in range(len(resx)):
      if resx[i] == ans:
        acc += 1
      tot += 1

    print(acc)
    print(tot)
    print(acc/tot)
    res.append(acc/tot)

  print(res)



if __name__ == "__main__":


  c0, c1, c2, c3, l = load_data()
  c0_train, c0_test = ttsplit(c0)
  c1_train, c1_test = ttsplit(c1)
  c2_train, c2_test = ttsplit(c2)
  c3_train, c3_test = ttsplit(c3)

  assert len(c0) == len(c1) == len(c2) == len(c3) == 2000
  assert len(l) == len(c0)*4

  k = 8
  bl = 4

  # for train
  t1 = time.time()
  c0t = pp(k, c0_train, bl, hashlib.blake2s)
  t2 = time.time()
  print(f'pp time: {t2-t1}')
  c1t = pp(k, c1_train, bl, hashlib.blake2s)
  c2t = pp(k, c2_train, bl, hashlib.blake2s)
  c3t = pp(k, c3_train, bl, hashlib.blake2s)


  x0 = pp(k, c0_test, bl, hashlib.blake2s)
  x1 = pp(k, c1_test, bl, hashlib.blake2s)
  x2 = pp(k, c2_test, bl, hashlib.blake2s)
  x3 = pp(k, c3_test, bl, hashlib.blake2s)

  res0 = test(x0, c0t, c1t, c2t, c3t)
  res1 = test(x1, c0t, c1t, c2t, c3t)
  res2 = test(x2, c0t, c1t, c2t, c3t)
  res3 = test(x3, c0t, c1t, c2t, c3t)
  print(res0)
  print(res1)
  print(res2)
  print(res3)
  


