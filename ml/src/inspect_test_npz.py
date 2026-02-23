import numpy as np, os
path = os.path.join("data","processed","test_data_1.npz")
data = np.load(path)
print("Keys:", data.files)
for k in data.files:
    arr = data[k]
    print(k, arr.shape, arr.dtype)