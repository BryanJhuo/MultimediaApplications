import numpy as np

f = open("../PythonProject/Homework/RegressionExample.txt", 'r').readlines()
y, x = [], []
for data in f:
    oneline = data.strip().replace("\n", "")
    oneline = oneline.split(" ")
    y.append(oneline[0])
    x.append(["1"] + oneline[1:5])

np_y = np.array(y).astype(int)
np_x = np.array(x).astype(int)
beta = np.dot(np.dot(np.linalg.inv(np.dot(np_x.T, np_x)), np_x.T), np_y)
print("Beta is : ", beta)
