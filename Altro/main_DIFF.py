
import os
import configparser
import torch 
import numpy as np 
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import  SGD
from model import Network, fit
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from ut import GiveMeSampleSize, BalanceSet, VALval, BalanceSetSynth
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

# Crea un oggetto ConfigParser
config = configparser.ConfigParser()

# Leggi il file di configurazione
config.read('/home/chiara/DataAUG/Altro/con.ini')

train_folder = config['AAA']['train_folder']
train_folder_Diff = config['AAA']['train_folder_Diff']
test_folder = config['AAA']['test_folder']
out_ss=len(os.listdir(train_folder))
in_ch=int(config['AAA']['in_ch'])
N_EPOCHS = int(config['AAA']['num_epoch'])
N = int(config['AAA']['N'])
BATCH_SIZE = int(config['AAA']['BATCH_SIZE'])
ss = int(config['AAA']['SS'])

drop = float(config['AAA']['DROP'])
lr = float(config['AAA']['LR'])

print(f'   \n input channel: {in_ch} \n output_channel: {out_ss}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



np.random.seed(0)
torch.manual_seed(0)
##############################################################
##########################  Sample size  #####################
##############################################################


sample_size_foody, sample_size_mine_1 = GiveMeSampleSize(alpha= 0.05, p=0.9, h=0.05,  out_ss = out_ss)

print(f'sample size foody: {sample_size_foody}, sample_size_mine: {sample_size_mine_1}')

sample_size_mine = sample_size_mine_1+ round(sample_size_mine_1*0.2)

transform =  transforms.Compose(
            [ transforms.ToTensor(),transforms.Grayscale(num_output_channels=1), transforms.Normalize((0.5), (1))])
test = ImageFolder(root=test_folder, transform=transform)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, num_workers=24)


def train_test_model(device , train_loader, val_loader,  N_EPOCHS, name, test_loader):
    model = Network(in_ch, out_ss, ss =ss, p = drop, up =0 ).to(device)
    
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    #Model
    writer = SummaryWriter(f'MODELLI/{name}')
    fit(device,model, train_loader, val_loader,criterion,optimizer,  N_EPOCHS, out_ss, writer, 0, name) 
    f1, cf= VALval(model, test_loader, out_ss,0)
    return f1,cf


#names = ['DIFF_40', 'DIFF_70', 'DIFF_90']
#per = [0.4, 0.7, 0.9]

names = ['DIFF_10', 'DIFF_20', 'DIFF_30','DIFF_50', 'DIFF_60', 'DIFF_80']
per = [0.1, 0.2, 0.3, 0.5, 0.6, 0.8]

for i, name  in enumerate(names):
    #Data

    train = BalanceSetSynth(root_path_real=train_folder,root_path_fake=train_folder_Diff, transform=transform, N = sample_size_mine, rate_fake=per[i])


    X_train, X_val = train_test_split(train,  test_size=0.2, random_state=3)
    train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, num_workers=24)
    val_loader = DataLoader(X_val, batch_size=BATCH_SIZE, num_workers=24)



    print('--- Starting trial: %s' % name)
    f1,cf = train_test_model(device , train_loader, val_loader,  N_EPOCHS, name, test_loader)
    print('--- End trial: %s, f1 TEST:' % name, f1)
    print('--- End trial: %s, cf TEST:' % name, cf)