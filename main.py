import argparse
from ctypes import util
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import CustomDataset
from preprocess import TimeseriesPreprocess
from model import convautoencoder, LSTMAutoencoder
from train import train
from utils import model_eval, model_save
import matplotlib.pyplot as plt
import argparse
import pickle
import random
import sys

random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

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
                    default=15, 
                    help= "choose proper epoch for training")
parser.add_argument("--model", 
                    required = False, 
                    type = str, 
                    default="lstmautoencoder", 
                    help= "choose proper model for training")

args = parser.parse_args()
writer = SummaryWriter()


def main(args):
  preprocess = TimeseriesPreprocess('./voucher.csv')
  preprocessed_data = preprocess.make_same_length_seq("id", "Item001", length = 320)
  dataset = CustomDataset(preprocessed_data)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
  # 모델 선정
  if args.model == "convautoencoder":
      model = convautoencoder(input_size= args.window_size)
  elif args.model == "lstmautoencoder":
      model = LSTMAutoencoder(seq_len= 320, n_features= 1)
  else:
      print("Wrong model input. Sys Stop!")
      sys.exit(0)
  # 옵티마이저와 적절한 에포크 설정
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
  nb_epochs = args.num_epoch
  # 학습 시작
  model = train(nb_epochs, dataloader, model, optimizer, writer) # 학습 시작
  
  # Model 저장
  PATH = './weights/'
  model_save(model = model, optimizer = optimizer, PATH= PATH)
  # Model 평가
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  model_eval(model, dataloader)


if __name__ == '__main__':
  main(args)

