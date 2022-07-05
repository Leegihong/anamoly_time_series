import torch
import torch.nn.functional as F
import torch.nn as nn

class convautoencoder(nn.Module):
       
    
    def __init__(self, input_size):
        super(convautoencoder, self).__init__() 
                
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels= 1, out_channels= 1 ,kernel_size= 8, stride=2, padding= 4),
            nn.ReLU(),
            nn.Conv1d(in_channels= 1, out_channels= 1 ,kernel_size= 8, stride=2, padding= 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels= 1, out_channels= 1, kernel_size= 8, stride=2, padding= 4),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels= 1, out_channels= 1 ,kernel_size=8, stride=2, padding= 4),
            nn.ReLU()
           
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_latent(self, x):
        self.encoder(x).size()
        return self.encoder(x)
    
