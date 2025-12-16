import os
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from PIL import Image
from UTgen import opt

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        
        self.l1 = nn.Linear(opt.latent_dim, 128 * self.init_size )
        self.norm1 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv1d(128, 128, 3, stride=1, padding=1)
        self.ELU = nn.ReLU( inplace=True)
        self.up = nn.Upsample(scale_factor=2)
        self.conv2 =    nn.Conv1d(128, 64, 3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm1d(64, 0.8)
        self.conv3 = nn.Conv1d(64, 28, 3, stride=1, padding=1)
        self.tan = nn.Tanh()
        

    def forward(self, z):
        # z dim [1, 128] = [input channel, latent_dim]
        out = self.l1(z) # [1, 128*init_size] = [1, 6272]
        out = out.view(out.shape[0], 128, self.init_size) # [1, 128, init_size**2]
        
        out = self.norm1(out)
        out = self.up(out)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.ELU(out)
        out = self.up(out)
        out = self.conv2(out) #torch.Size([1, 64, 196])
        out = self.norm3(out)
        #out = self.up(out)
        out = self.conv3(out)
        img = self.tan(out)
        img = img.view(out.shape[0], 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 3, 2, 1),
            nn.BatchNorm1d(128, 0.8),
            nn.Dropout(0.25),
            nn.ReLU( inplace=True),
            nn.Conv1d(128, 256, 3, 2, 1),
            nn.BatchNorm1d(256, 0.8),
            nn.Dropout(0.25),
            nn.ReLU( inplace=True),
            nn.Conv1d(256, 512, 3, 2, 1),
            nn.BatchNorm1d(512, 0.8),
            nn.Dropout(0.25),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(25088, 1),
            nn.Sigmoid()
        )


    def forward(self, img):
        
        input= img.view(img.shape[0], 1, img.shape[2]*img.shape[3])
        validity = self.model(input)
        return validity








def generate_and_save_images(imgs, model, epoch, test_input, classd):
    predictions = []
    for _ in range(0,test_input):
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = model(z)
        predictions.append(gen_imgs)

    for ii,i in enumerate(predictions):
        for idx, t in enumerate(i):
            
            new_img = Image.fromarray(t.to('cpu').detach().numpy().squeeze(), mode='L')  # Squeeze the singleton dimension

            # Save the new image as TIFF
            try:
                new_img.save('/home/chiara/DataAUG/DATA/MAL_dataset/GAN/{}/image_at_epoch1_{:04d}-{}_{}.tiff'.format(classd,epoch, idx, ii))
            except: 
                os.mkdir(f'/home/chiara/DataAUG/DATA/MAL_dataset/GAN/{classd}/')
                new_img.save('/home/chiara/DataAUG/DATA/MAL_dataset/GAN/{}/image_at_epoc1_{:04d}-{}_{}.tiff'.format(classd,epoch, idx, ii))

def generate_and_save_images2( model, epoch, test_input, classd):
    predictions = []
    for _ in range(0,test_input):
        #z = Variable(Tensor(np.random.normal(0, 1, (28, opt.latent_dim))))
        z = torch.tensor(np.random.normal(0, 1, (28, opt.latent_dim)), dtype=torch.float, device='cuda')

        # Generate a batch of images
        gen_imgs = model(z)
        predictions.append(gen_imgs)

    for ii,i in enumerate(predictions):
        for idx, t in enumerate(i):
            
            new_img = Image.fromarray(t.to('cpu').detach().numpy().squeeze(), mode='L')  # Squeeze the singleton dimension

            dir = '/home/chiara/DataAUG/DATA/MAL_dataset/GAN'
            # Save the new image as TIFF
            try:
                new_img.save('{}/{}/simage_at_epoch1_{:04d}-{}_{}.tiff'.format(dir,classd,epoch, idx, ii))
            except: 
                os.mkdir(f'{dir}/{classd}/')
                new_img.save('{}/{}/simage_at_epoch1_{:04d}-{}_{}.tiff'.format(dir,classd,epoch, idx, ii))

