import torch
import numpy as np



def model_save(model, optimizer, PATH):
    # Model 저장
    print("Start saving the model")

    PATH = './weights/'

    torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
    torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, PATH + 'all.tar')

    print("End saving the model")
    
def model_eval(model, dataloader):
    model.eval()
    train_mae_loss = []
    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            x_train, y_train = samples
            
            test_prediction = model(x_train)

            loss = np.mean(np.abs(test_prediction.detach().numpy().flatten() - y_train.detach().numpy().flatten()))
            train_mae_loss.append(loss)
    
    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold: ", threshold)