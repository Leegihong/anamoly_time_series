import torch
from model import convautoencoder


PATH = './weights/'

model = convautoencoder(288)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 


model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장

checkpoint = torch.load(PATH + 'all.tar')   # dict 불러오기
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

model.eval()

def predict(model, dataset,batch_size=1):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    dataset_ae = AutoencoderDataset(dataset)
    dataloader_ae = DataLoader(dataset_ae, 
                               batch_size=batch_size,
                               shuffle=False,num_workers=8)
    with torch.no_grad():
        model = model.eval()
        if batch_size == len(dataset) :
            seq_true  =next(dataloader_ae.__iter__())
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
        else :
            for idx , seq_true in enumerate(dataloader_ae):
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                predictions.append(seq_pred.cpu().numpy().flatten())
                losses.append(loss.item())
    return predictions, losses