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
import matplotlib.pyplot as plt
import argparse
import pickle
import random


def train(nb_epochs, dataloader, model, optimizer, writer):
    cost_list = []
    # 학습 시작
    for epoch in range(nb_epochs + 1):
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            prediction = model(x_train)

            # cost 계산
            cost = F.mse_loss(prediction, y_train)
            cost_list.append(cost.item())

            optimizer.zero_grad()

            cost.backward()
            
            optimizer.step()
            
            writer.add_scalar('total loss', cost, epoch)
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs, batch_idx+1, len(dataloader), cost.item()))
    
    
    # 학습 완료 
    print("Training is over!\n")
    last_cost = cost_list[-1]
    print(f"Last of cost = {last_cost}")
    
    return model