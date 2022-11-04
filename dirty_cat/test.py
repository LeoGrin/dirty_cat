from tests.test_minhash_encoder import profile_encoder
from dirty_cat import MinHashEncoder
import time
from dirty_cat.datasets import fetch_employee_salaries
import numpy as np


employee_salaries = fetch_employee_salaries()
df = employee_salaries.X
X = df[["employee_position_title"]]
print("X.shape", X.shape)
print(X.shape)
enc = MinHashEncoder(n_components=200, hashing="fast", minmax_hash=True, n_jobs=-1)

time_batch = []
time_non_batch = []
for i in range(10):
    t0 = time.time()
    enc.fit(X)
    y1 = enc.transform(X, batch=True)
    time_batch.append(time.time() - t0)
    t0 = time.time()
    enc.fit(X)
    y2 = enc.transform(X, batch=False)
    time_non_batch.append(time.time() - t0)
    assert np.allclose(y1, y2)

print(time_batch)
print(time_non_batch)

print(np.mean(time_batch))
print(np.mean(time_non_batch))