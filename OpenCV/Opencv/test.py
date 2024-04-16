import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.preprocessing import LabelEncoder


# define model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


# set torch
torch.manual_seed(202404)
enable_cuda = False
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# read data
dataframe = pd.read_excel('data_weight_multlfeatutes.xlsx')

# clean data
dataframe["性別"] = LabelEncoder().fit_transform(dataframe["性別"])
dataframe["身高"] = dataframe["身高"] / 200
dataframe["體重"] = dataframe["體重"] / 100
data = torch.FloatTensor(np.array(dataframe[['身高', '體重']]))
target = torch.FloatTensor(np.array(dataframe['性別'])).view(-1, 1)
dataset = torch.utils.data.TensorDataset(data, target)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=data.shape[0], shuffle=True)

# define model
model_1 = Model().to(device)
model_2 = Model().to(device)

# define loss
loss = torch.nn.BCELoss().to(device)

# define optimizer
optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=0.1, momentum=0.9)
optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=0.01, momentum=0.9)

losses_1 = []
losses_2 = []

# train model
model_1.train()
for epoch in range(10000):
    now_loss = None
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        output = model_1(data)
        now_loss = loss(output, target)
        optimizer_1.zero_grad()
        now_loss.backward()
        optimizer_1.step()
    losses_1.append(now_loss.item())

model_2.train()
for epoch in range(10000):
    now_loss = None
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        output = model_2(data)
        now_loss = loss(output, target)
        optimizer_2.zero_grad()
        now_loss.backward()
        optimizer_2.step()
    losses_2.append(now_loss.item())

# print bias
print("learning rate = 0.1:")
print(f"beta_0 = {model_1.linear.bias.item():.4f}")
print(f"beta_1 = {model_1.linear.weight[0][0].item():.4f}")
print(f"beta_2 = {model_1.linear.weight[0][1].item():.4f}")
print("learning rate = 0.01:")
print(f"beta_0 = {model_2.linear.bias.item():.4f}")
print(f"beta_1 = {model_2.linear.weight[0][0].item():.4f}")
print(f"beta_2 = {model_2.linear.weight[0][1].item():.4f}")

# plot loss
plt.figure()
plt.plot(losses_1, label="loss = 0.1", color="red")
plt.plot(losses_2, label="loss = 0.01", color="blue")
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
    output = model_1(grids)
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
    output = model_2(grids)
    axis.contourf(grid_x, grid_y, output.reshape(grid_x.shape), cmap="coolwarm", levels=500)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap="seismic")
plt.title("learning rate = 0.01")
plt.colorbar()
plt.show()
