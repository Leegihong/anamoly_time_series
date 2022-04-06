import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CustomDataset
from model import convautoencoder


dataset = CustomDataset('.\small_noise.csv')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = convautoencoder(input_size= 288)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
nb_epochs = 20

# 학습 시작
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))