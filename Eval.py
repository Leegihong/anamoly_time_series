from cgi import test
import torch
from model import convautoencoder
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import CustomDataset
import matplotlib.pyplot as plt
import numpy as np
from preprocess import TimeseriesPreprocess
import pickle

PATH = './weights/'
threshold = 0.02695318
model = convautoencoder(288)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 


model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

checkpoint = torch.load(PATH + 'all.tar')   # dict 불러오기
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])



with torch.no_grad():
    pred = {}
    lat = {}
    origin = {}
    test_mae_loss = []
    preprocess = TimeseriesPreprocess('./data_wave.csv')
    preprocessed_data = preprocess.make_same_length_seq("WaveId", "Item001", length= 320)
    dataset = CustomDataset(preprocessed_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            x_test, y_test = samples
            
            test_prediction = model(x_test)
            latent = model.get_latent(x_test)
            
            pred[batch_idx+1] = test_prediction.cpu().numpy().flatten()
            lat[batch_idx+1] = latent.cpu().numpy().flatten()
            origin[batch_idx+1] = y_test.cpu().numpy().flatten()
            
            loss = np.mean(np.abs(test_prediction.detach().numpy().flatten() - y_test.detach().numpy().flatten()))
            test_mae_loss.append(loss)


plt.hist(test_mae_loss)
plt.show()

anomalies = [index for index, ano in enumerate(test_mae_loss) if ano > threshold]
anomal = {}
for index, ano in enumerate(test_mae_loss):
    if ano > threshold:
        anomal[index+1] = ano

# print(anomalies)
print("Number of anomaly samples: ", len(anomalies))
print("Indices of anomaly samples: ", anomalies)
print("Values of anomaly :", [test_mae_loss[i] for i in anomalies])

with open('result_dict.pkl','wb') as f:
    pickle.dump(pred,f)
with open('origin_dict.pkl','wb') as f:
    pickle.dump(origin,f)
with open('data_dict.pkl','wb') as f:
    pickle.dump(lat,f)
with open('anomaliy_dict.pkl','wb') as f:
    pickle.dump(anomal,f)



            





















































# def predict(model, dataset,batch_size=4):
#     predictions, losses = [], []
#     criterion = F.L1Loss(reduction='sum')
#     dataset_ae = CustomDataset('./daily_jumpsup.csv')
#     dataloader_ae = DataLoader(dataset_ae, 
#                                batch_size=batch_size,
#                                shuffle=False)
#     with torch.no_grad():
#         model = model.eval()
#         if batch_size == len(dataset) :
#             seq_true  =next(dataloader_ae.__iter__())
#             seq_true = seq_true
#             seq_pred = model(seq_true)
#             loss = criterion(seq_pred, seq_true)
#             predictions.append(seq_pred.cpu().numpy().flatten())
#             losses.append(loss.item())
#         else :
#             for idx , seq_true in enumerate(dataloader_ae):
#                 seq_true = seq_true
#                 seq_pred = model(seq_true)
#                 loss = criterion(seq_pred, seq_true)
#                 predictions.append(seq_pred.cpu().numpy().flatten())
#                 losses.append(loss.item())
#     return predictions, losses