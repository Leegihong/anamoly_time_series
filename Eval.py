import torch
from model import convautoencoder
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import CustomDataset
import matplotlib.pyplot as plt
import numpy as np

PATH = './weights/'
threshold = 0.9
model = convautoencoder(288)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 


model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

checkpoint = torch.load(PATH + 'all.tar')   # dict 불러오기
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])



with torch.no_grad():
    pred = []
    test_mae_loss = []
    dataset = CustomDataset('./daily_jumpsup.csv')
    dataloader = DataLoader(dataset, batch_size = 4, shuffle= False)
    model.eval()
    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            x_test, y_test = samples
            
            test_prediction = model(x_test)
            pred.append(test_prediction.cpu().numpy().flatten())

            loss = np.mean(np.abs(test_prediction.detach().numpy().flatten() - y_test.detach().numpy().flatten()))
            test_mae_loss.append(loss)

plt.hist(test_mae_loss, bins= 50)
plt.xlim([0,1])
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

anomalies = [index for index, ano in enumerate(test_mae_loss) if ano > threshold]
# print(anomalies)
print("Number of anomaly samples: ", len(anomalies))
print("Indices of anomaly samples: ", anomalies)
print("Values of anomaly :", [test_mae_loss[i] for i in anomalies])



            





















































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