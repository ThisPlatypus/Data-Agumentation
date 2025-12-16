import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from UTgen import  SingleClassDataset
from Model import Generator, Discriminator, weights_init_normal, generate_and_save_images
from UTgen import opt

cuda = True if torch.cuda.is_available() else False

fold = '/home/chiara/DataAUG/DATA/MAL_dataset/TR_HEX'

for classd in os.listdir(fold):
    print(classd)

    
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    image_folder = '/home/chiara/DataAUG/DATA/MAL_dataset/HEX/mirai' #f'{fold}/{classd}/'
    
    transform =  transforms.Compose(
            [ transforms.ToTensor(),transforms.Grayscale(num_output_channels=1), transforms.Normalize((0.5), (1))])
    
    # Creating dataset
    dataset = SingleClassDataset(image_folder, transform)
    
    # Creating dataloader
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    
    optimizer_G = torch.optim.Adagrad(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adagrad(discriminator.parameters(), lr=opt.lr)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # ----------
    #  Training
    # ----------
    writer = SummaryWriter(f'/home/chiara/DataAUG/RESULT/GSN/{classd}')
    for epoch in range(opt.n_epochs):
        for i, imgs in enumerate(dataloader):
            
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            
            writer.add_scalar('generator loss',
                g_loss,
                epoch + i)
            
            writer.add_scalar('real loss',
                real_loss,
                epoch + i)
            
            writer.add_scalar('fake loss',
                fake_loss,
                epoch + i)
            writer.add_scalar('discriminator loss',
                d_loss,
                epoch + i)

            batches_done = epoch * len(dataloader) + i
            

    #generate_and_save_images(imgs, generator, epoch, 100, classd)

    torch.save( generator.state_dict(), f'/home/chiara/DataAUG/MOD/GAN/{classd}_.pth')

    writer.close()
    break
print('FINISH')