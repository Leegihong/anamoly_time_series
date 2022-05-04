import torch
import torch.nn.functional as F
import torch.nn as nn

class convautoencoder(nn.Module):
       
    
    def __init__(self, input_size):
        super(convautoencoder, self).__init__() 
        
        self.fc1 = nn.Conv1d(in_channels= 1, out_channels= 32 ,kernel_size= 7)
        self.fc2 = nn.Conv1d(in_channels= 32, out_channels= 16 ,kernel_size= 7)
        self.fc3 = nn.ConvTranspose1d(in_channels= 16, out_channels= 32 ,kernel_size= 7)
        self.fc4 = nn.ConvTranspose1d(in_channels= 32, out_channels= 1 ,kernel_size= 7)  
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x
    
    
    
    
    
    
    
#     self.encoder = nn.Sequential(
#         nn.Conv1d(1, 16, 5, stride = 10, padding  = 1),  # N, 16, 1639
#         nn.ReLU(),
#         nn.Conv1d(16, 32, 9, stride = 10, padding = 0), # N, 32, 164
#         nn.ReLU(),
#         nn.Conv1d(32, 64, 5, stride = 10, padding = 1), # N, 64, 17
#         nn.ReLU(),
#         nn.Conv1d(64, 128, 9, stride = 10, padding = 2), # N, 128, 2
#     )

#     # N, 64, 1, 1
#     self.decoder = nn.Sequential(
#         nn.ConvTranspose1d(128, 64, 9, stride = 10, padding = 2), # N, 64, 17
#         nn.ReLU(),
#         nn.ConvTranspose1d(64, 32, 5, stride = 10, padding = 1), # N, 32, 164 
#         nn.ReLU(),
#         nn.ConvTranspose1d(32, 16, 9, stride = 10, padding = 0), # N, 16, 1639
#         nn.ReLU(),
#         nn.ConvTranspose1d(16, 1, 5, stride = 10, padding = 1), # N, 1, 16384 
#         nn.ReLU()
#     )
  
#   def forward(self, x):
#     encoded = self.encoder(x)
#     decoded = self.decoder(encoded)
#     return decoded