import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

torch.manual_seed(0)

class convautoencoder(nn.Module):
       
    
    def __init__(self, input_size):
        super(convautoencoder, self).__init__() 
        self.kernel_size = 6
        self.padding_size = 4
                
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels= 1, out_channels= 4 ,kernel_size= self.kernel_size, stride=2, padding= self.padding_size),
            nn.ReLU(),
            nn.Conv1d(in_channels= 4, out_channels= 8 ,kernel_size= self.kernel_size, stride=2, padding= self.padding_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels= 8, out_channels= 4, kernel_size= self.kernel_size, stride=2, padding= self.padding_size),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels= 4, out_channels= 1 ,kernel_size=self.kernel_size, stride=2, padding= self.padding_size),
            nn.ReLU()
           
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_latent(self, x):
        self.encoder(x).size()
        return self.encoder(x)
     


### LSTM-AE ######
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = (
            embedding_dim, 2 * embedding_dim
        )
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=1,
          batch_first=True
        )
    def forward(self, x):
        x = torch.reshape(x, (x.size()[0], self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return  x[:,-1,:]
    
    
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )
        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)
        self.timedist = TimeDistributed(self.output_layer, batch_first= True)
        
    def forward(self, x):
        x=x.reshape(-1,1,self.input_dim).repeat(1,self.seq_len,1)       
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        return self.timedist(x)


    
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)#.to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)#.to(device)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x