from msilib import sequence
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset): 
  def __init__(self, data):
      # self.x_data = pd.read_csv(path).iloc[:, 1].values
      self.x_data = data
      self.train_data_tensor = torch.FloatTensor(self.x_data)
      self.train_data_final = torch.reshape(self.train_data_tensor, (self.train_data_tensor.shape[0], self.train_data_tensor.shape[1], 1)) # lstm autoencoder
      # self.train_data_final = torch.reshape(self.train_data_tensor, (self.train_data_tensor.shape[0], 1, self.train_data_tensor.shape[1])) # convolution atuoencoder
      # self.train_data_final = self.train_data_tensor

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.train_data_final)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = self.train_data_final[idx]
    y = self.train_data_final[idx]
    return x, y
 
  
            
  
                
      