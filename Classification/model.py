#MODEL
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ut import VALval
from tqdm import tqdm, trange
import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Define a convolution neural network
class Network(nn.Module):
    def __init__(self, in_ch,out_ss,  ss, p, up):
        super(Network, self).__init__()

        if up == 1:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        
        self.conv1 = nn.Conv1d(in_channels= in_ch, out_channels= ss*2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels= ss*2, out_channels=ss*4, kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(ss*4)
        self.conv3 = nn.Conv1d(in_channels=ss*4, out_channels=ss*2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)

        
        if ss == 16 and up ==0:
            self.fc1 = nn.Linear(25088, ss*4) #48*48*32*8) #dense - got the 17 logits
        elif ss == 16 and up ==1:
            self.fc1 = nn.Linear(100352, ss*4)
        elif ss == 8 and up ==0:
            self.fc1 = nn.Linear(12544, ss*4)
        elif ss == 32 and up ==0:
            self.fc1 = nn.Linear(25088*2, ss*4)
        
        self.dropout = nn.Dropout(p=p)
        self.fc2 = nn.Linear(ss*4, ss)
        
        self.fc3 = nn.Linear(ss,out_ss)


    def forward(self, input, flag):
        if flag==1:
            input =self.upsample(input)
            
        output =  input.view(input.shape[0], 1, input.shape[2]*input.shape[3])
        output = F.relu(self.conv1(output))
        #output = self.dropout(output)
        output = F.relu(self.conv2(output))
        
        
        output = self.bn1(output)
        
        output = F.relu(self.conv3(output))
        
        
        output = self.dropout(output)

        output = output.reshape(-1,output.shape[1]*output.shape[2]) #flatten considering the batch size output.shape[0] and other dimension
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)


        return output




def fit(device, model, train_loader, val_loader,criterion,optimizer,  epochs, out_ss, writer, flag, name):
    #(model, train_loader, val_loader, epochs=N_EPOCHS) # Run with 1 epoch to speed things up for 
    best_f1=0
    for epoch in trange(epochs, desc="Training"):
        train_loss = 0.0
        for x, y  in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            # aggiungi che deve paddare
            x, y = x.to(device), y.to(device)
            
            
            y_hat = model(x, flag)
            
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val_f1, _ = VALval(model, val_loader,out_ss, flag)
            if best_f1< val_f1:
                torch.save(model.state_dict(), f'/home/chiara/DataAUG/Classification/MODELLI/{name}.pth')
        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.5f}")
        print(f"\n Epoch {epoch + 1}/{epochs} f1 val: {val_f1:.5f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("F1/VAL", val_f1, epoch)



