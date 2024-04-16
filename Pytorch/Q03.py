import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import torch
import torch.utils.data

# define model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

def load_classification_example():
    filename = './PythonProject/Homework/data_weight_multlfeatutes.xlsx'
    df = pd.read_excel(filename)
    labelencoder = LabelEncoder()
    df["性別"] = labelencoder.fit_transform(df["性別"])
    df["手機品牌"] = labelencoder.fit_transform(df["手機品牌"])
    X = np.array(df[['身高', '體重']])
    Y = np.array(df['性別'])
    return X, Y

torch.manual_seed(202404)
enable_cuda = False
device = torch.device("cuda" if torch.cuda.is_available() and enable_cuda else "cpu")

dataframe, target = load_classification_example()
dataframe[: ,0] = dataframe[: ,0] / 200
dataframe[: ,1] = dataframe[: ,1] / 100

data = torch.FloatTensor(dataframe)
target = torch.FloatTensor(target).view(-1, 1)
dataset = torch.utils.data.TensorDataset(data, target)
dataloader = torch.utils.data.DataLoader(dataset, batch_size= data.shape[0], shuffle=True)
# model
model1 = Model().to(device)
model2 = Model().to(device)
# loss
loss = torch.nn.BCELoss().to(device)
# optimizer
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1, momentum=0.9)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)

loss_model_1 = []
loss_model_2 = []
# train
model1.train()
for epoch in range(10000):
    nowLoss = None
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        output1 = model1(data)
        nowLoss = loss(output1, target)
        optimizer1.zero_grad()
        nowLoss.backward()
        optimizer1.step()
    loss_model_1.append(nowLoss.item())

model2.train()
for epoch in range(10000):
    nowLoss = None
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        output2 = model2(data)
        nowLoss = loss(output2, target)
        optimizer2.zero_grad()
        nowLoss.backward()
        optimizer2.step()
    loss_model_2.append(nowLoss.item())

print("learning rate is 0,1 :")
print(f"beta_0 = {model1.linear.bias.item(): .4f}")
print(f"beta_1 = {model1.linear.weight[0][0].item(): .4f}")
print(f"beta_2 = {model1.linear.weight[0][1].item(): .4f}")

print("learning rate is 0.01:")
print(f"beta_0 = {model2.linear.bias.item(): .4f}")
print(f"beta_1 = {model2.linear.weight[0][0].item(): .4f}")
print(f"beta_2 = {model2.linear.weight[0][1].item(): .4f} ")

'''
# plot loss
plt.figure()
plt.plot(loss_model_1, label="loss = 0.1", color="red")
plt.plot(loss_model_2, label="loss = 0.01", color="blue")
plt.legend()
plt.show()

# plot decision boundary
x_min = 0.7
x_max = 1.0
y_min = 0.4
y_max = 1.0
grid_size = 0.03

grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max + grid_size, grid_size),
                             np.arange(y_min, y_max + grid_size, grid_size))
grids = np.column_stack((grid_x.ravel(), grid_y.ravel()))
grids = torch.FloatTensor(grids).to(device)

plt.figure()
axis = plt.gca()
axis.set_xlim([x_min, x_max])
axis.set_ylim([y_min, y_max])
with torch.no_grad():
    output = model1(grids)
    axis.contourf(grid_x, grid_y, output.reshape(grid_x.shape), cmap="coolwarm", levels=500)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap="seismic")
plt.title("learning rate = 0.1")
plt.colorbar()
plt.show()

plt.figure()
axis = plt.gca()
axis.set_xlim([x_min, x_max])
axis.set_ylim([y_min, y_max])
with torch.no_grad():
    output = model2(grids)
    axis.contourf(grid_x, grid_y, output.reshape(grid_x.shape), cmap="coolwarm", levels=500)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap="seismic")
plt.title("learning rate = 0.01")
plt.colorbar()
plt.show()

'''