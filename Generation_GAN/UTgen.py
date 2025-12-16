import argparse
import os
from PIL import Image
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=20, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=150, help="interval between image sampling")
opt = parser.parse_args()




class SingleClassDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        """
        Args:
            image_folder (string): Path to the folder where images are.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.image_files =[]
        image_paths= [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        if len(image_paths) > (800):
                image_paths = image_paths[1:800]
                for image_path in image_paths:
                    self.image_files.append(image_path)
                    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image
   

    

