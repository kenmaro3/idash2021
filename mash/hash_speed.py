import time
import numpy as np
import hashlib

t1 = "AGCT"
t2 = "AACG"


print("sha256")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha256(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))
#print(hashlib.sha256(t1.encode()).hexdigest()[:8 ]) # 32-bit, 8  hex chars
#start = time.time()
#print(hashlib.sha256(t1.encode()).hexdigest()[:16]) # 64-bit, 16 hex chars
#end = time.time()
#time2 = end-start
#print(hashlib.sha256(t1.encode()).hexdigest()[:16]) # 64-bit, 16 hex chars

print("blake2b")
times = []
for i in range(100):
  start = time.time()
  hashlib.blake2b(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("blake2s")
times = []
for i in range(100):
  start = time.time()
  hashlib.blake2s(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("md5")
times = []
for i in range(100):
  start = time.time()
  hashlib.md5(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("sha1")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha1(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("sha224")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha224(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("sha256")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha256(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))


print("sha384")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha384(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("sha3_224")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha3_224(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("sha3_256")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha3_256(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("sha3_384")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha3_384(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("sha3_512")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha3_512(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

print("sha512")
times = []
for i in range(100):
  start = time.time()
  hashlib.sha512(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
  end = time.time()
  time1 = end-start
  times.append(time1)
print(np.average(times))

#print("shake_128")
#times = []
#for i in range(100):
#  start = time.time()
#  hashlib.shake_128(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
#  end = time.time()
#  time1 = end-start
#  times.append(time1)
#print(np.average(times))

#print("shake_256")
#times = []
#for i in range(100):
#  start = time.time()
#  hashlib.shake_256(t1.encode()).hexdigest()[:8 ] # 32-bit, 8  hex chars
#  end = time.time()
#  time1 = end-start
#  times.append(time1)
#print(np.average(times))
