import os
import torch
from Model import Generator, weights_init_normal, generate_and_save_images2

cuda = True if torch.cuda.is_available() else False

fold = '/home/chiara/DataAUG/DATA/MAL_dataset/TR_HEX'

for classd in os.listdir(fold):
    print(classd)
    
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    generator.load_state_dict(torch.load(f'/home/chiara/DataAUG/MOD/GAN/{classd}_.pth'))
    if cuda:
        generator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    
    generate_and_save_images2(generator, 99, 700, classd)
    
print('FINISH')