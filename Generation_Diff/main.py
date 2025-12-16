import random
import numpy as np
from argparse import ArgumentParser
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import  Dataset, DataLoader
import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Lambda
from Mode import MyDDPM, MyUNet, training_loop 
from ut import SingleClassDataset
# Setting reproducibility
SEED = 0
SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper parameters
no_train = False
fashion = True
batch_size = 128
n_epochs = 500
lr = 0.001



fold = '/home/chiara/DataAUG/DATA/MAL_dataset/TR_HEX'

for classd in os.listdir(fold):
    print(classd)
    
    image_folder = f'{fold}/{classd}/'
    
    transform =  transforms.Compose(
            [ transforms.ToTensor(),transforms.Grayscale(num_output_channels=1), transforms.Normalize((0.5), (1))])
    
    # Creating dataset
    dataset = SingleClassDataset(image_folder, transform=transform) 
    writer = SummaryWriter(f'/home/chiara/DataAUG/RESULT/DIFF/{classd}')
    # Creating dataloader
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    n_steps, min_beta, max_beta = 1000, 10 ** -4, 0.02  # Originally used by the authors
    ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

    training_loop(writer, ddpm, dataloader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=f'/home/chiara/DataAUG/MOD/DIFF/{classd}.pth')
    
