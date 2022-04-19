import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset): 
  def __init__(self, path):
    self.x_data = pd.read_csv(path).iloc[:, 1].values
    self.train_data = self.make_sequence(self.x_data, 288)
    self.train_data_tensor = torch.FloatTensor(self.train_data)
    self.train_data_final = torch.reshape(self.train_data_tensor, (self.train_data_tensor.shape[0], 1, self.train_data_tensor.shape[1]))
    # self.y_data = [[152], [185], [180], [196], [142]]

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.train_data_final)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.train_data_final[idx])
    # y = torch.FloatTensor(self.y_data[idx])
    return x
 
  def make_sequence(self, data, size):
      seq = []
      for i in range(0, len(data) - size):
        seq.append(data[i :i+size])
      return seq
            
      
