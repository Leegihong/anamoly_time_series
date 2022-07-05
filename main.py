import argparse
from ctypes import util
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CustomDataset
from preprocess import TimeseriesPreprocess
from model import convautoencoder
import matplotlib.pyplot as plt
import argparse
import pickle
import random

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

def model_save(model, optimizer, PATH):
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
    

    


parser = argparse.ArgumentParser(description= "input parameters")

parser.add_argument("--batch_size", 
                    type = int, 
                    default=16,
                    help= "choose proper batch size for training")
parser.add_argument("--window_size", 
                    required = False, 
                    type = int, 
                    default=288, 
                    help= "choose proper sequence size for building model")
parser.add_argument("--num_epoch", 
                    required = False, 
                    type = int, 
                    default=40, 
                    help= "choose proper epoch for training")

args = parser.parse_args()
writer = SummaryWriter()


def main(args):
  preprocess = TimeseriesPreprocess('./voucher.csv')
  preprocessed_data = preprocess.make_same_length_seq("id", "Item001")
  dataset = CustomDataset(preprocessed_data)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
  model = convautoencoder(input_size= args.window_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
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

  # # Model 저장
  dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
  PATH = './weights/'
  model_save(model = model, optimizer = optimizer, PATH= PATH)

  model.eval()
  train_mae_loss = []
  with torch.no_grad():
      for batch_idx, samples in enumerate(dataloader):
          x_train, y_train = samples
          
          test_prediction = model(x_train)

          loss = np.mean(np.abs(test_prediction.detach().numpy().flatten() - y_train.detach().numpy().flatten()))
          train_mae_loss.append(loss)
  
  # Get reconstruction loss threshold.
  threshold = np.max(train_mae_loss)
  print("Reconstruction error threshold: ", threshold)
          
  

  
if __name__ == '__main__':
  main(args)

