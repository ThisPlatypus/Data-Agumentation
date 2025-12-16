from torchvision.datasets import DatasetFolder
import os
from PIL import Image
import os
from torch.utils.data import Dataset

class SingleClassDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        """
        Args:
            image_folder (string): Path to the folder where images are.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image
   
