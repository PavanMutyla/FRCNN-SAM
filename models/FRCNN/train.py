import torch
from .frcnn import model
import torch.optim
import time
import data_prep
from tqdm.auto import tqdm
import os

def train(data_loader : torch.utils.data.DataLoader, save_path:str, device=str, epochs = 10, n = int,model = model): # default device 'cpu'
    '''
    Train the faster r-cnn model and returns the metrics and saves the trained weights in the give save_path, which is used in the final dratf 

    ARGS:
    model: called form frcnn.py
    data: pre-annotated data 
    n: no of classes without background.
    '''
    model = model.to(device=device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.001, momentum=0.9)
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features=1024, out_features=n+1)
    train_loss = 0
    start = time.time()
    for epoch in range(tqdm(epochs)):
        for image, target in data_loader:
            images = [img.to(device) for img in image]
            targets = [tar.to(device) for tar in target]
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss+=loss
        print(f'loss:{train_loss/len(data_loader)}')
    
    end = time.time()
    time_taken = end-start
    print(f'time taken to train {time_taken}')
    save_dir = os.path.join(save_path, 'faster_rcnn_weigths_trained.pth')
    torch.save(model.state_dict(), save_dir)
    print(f'saved model at {save_dir}')







