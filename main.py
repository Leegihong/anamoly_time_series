import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CustomDataset
from model import convautoencoder
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description= "input parameters")

parser.add_argument("--batch_size", type = int, default=4,help= "choose proper batch size for training")
parser.add_argument("--window_size", required = False, type = int, default=288, help= "choose proper sequence size for building model")
parser.add_argument("--num_epoch", required = False, type = int, default=20, help= "choose proper epoch for training")

args = parser.parse_args()

writer = SummaryWriter()

def main(args):
  dataset = CustomDataset('.\small_noise.csv')
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
  model = convautoencoder(input_size= args.window_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
  nb_epochs = args.num_epoch
  cost_list = []
  # 학습 시작
  for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
      x_train, y_train = samples

      prediction = model(x_train)

      # cost 계산
      cost = F.mse_loss(prediction, y_train)
      cost_list.append(cost.item())

      # gradient를 0으로 초기화 -> 이전에 계산된 gradient값이 남아있어서 그것을 초기화하기 위하여 (pytorch 특징)
      optimizer.zero_grad()
      # cost function을 미분하여 gradient를 계산
      cost.backward()
      # layer에 있는 parameters를 업데이트
      optimizer.step()
    writer.add_scalar('total loss', cost, epoch)
    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))
    
    
  # 학습 완료 
  print("Training is over!\n")
  last_cost = cost_list[-1]
  print(f"Last of cost = {last_cost}")

  # Model 저장
  print("Start saving the model")

  PATH = './weights/'

  torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
  torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
  torch.save({
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict()
  }, PATH + 'all.tar')

  print("End saving the model")

  model.eval()
  x_train_pred = model(x_train)

  train_mae_loss = np.mean(np.abs(x_train_pred.detach().numpy() - y_train.numpy()), axis=1)
  print(train_mae_loss)
  plt.hist(train_mae_loss)
  plt.xlim([0,1])
  plt.xlabel("Train MAE loss")
  plt.ylabel("No of samples")
  plt.show()

  # Get reconstruction loss threshold.
  threshold = np.max(train_mae_loss)
  print("Reconstruction error threshold: ", threshold)
  
if __name__ == '__main__':
  main(args)

