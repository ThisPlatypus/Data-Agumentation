import statsmodels.stats.power as smp
import os
from scipy.stats import norm
from torchvision.datasets import DatasetFolder
from PIL import Image
from torchmetrics.classification import MulticlassF1Score
import torch
from sklearn.metrics import confusion_matrix
import numpy as np

# TEST function

def calculate_sample_size(alpha, p, h):
    """Calculate sample size for estimating population proportion.
    
    Args:
    alpha (float): Significance level (e.g., 0.05 for 95% confidence level).
    p (float): Expected population proportion (between 0 and 1).
    h (float): Margin of error (half the width of the confidence interval).
    
    Returns:
    float: Sample size required.
    """
    # Calculate z_alpha from the significance level alpha
    z_alpha = norm.ppf(1 - alpha/2)  # Two-tailed test
    
    # Calculate sample size
    n = (z_alpha ** 2 * p * (1 - p)) / (h ** 2)
    return round(n)


def GiveMeSampleSize(alpha, p, h,  out_ss):
    # calculating n from Foody
    sample_size_foody = calculate_sample_size(alpha, p, h)
    sample_size = round(sample_size_foody/out_ss)
    
    d = 0.15 # Small effect: d = 0.2 Medium effect: d = 0.5 Large effect: d = 0.8
    
    print(f'D: {d}')
        # Define parameters
    alpha = 0.05  # desired significance level
    power = (1-0.128)  # desired statistical power
    effect_size = d # desired effect size
    num_comparisons = out_ss  # number of pairwise comparisons

    # Perform power analysis
    sample_size_mine = smp.GofChisquarePower().solve_power(effect_size=effect_size, power=power, alpha=alpha, n_bins=num_comparisons)

    return sample_size_foody, round(sample_size_mine)




class BalanceSet(DatasetFolder):
    def __init__(self, root_path, transform=None, N=100):
        self.root_path = root_path
        self.transform = transform
        self.N = N  # Number of examples to keep for class 10
        self.image_paths = []
        self.labels = []


        self.label_mapping = {
            'hydra_ssh2': 0,
            'java_rmi': 1,
            'mirai': 2,
            'netbios_ssn2': 3,
            'replayAttacks': 4
            }

        # Get a list of folder names (classes)
        folder_names = sorted(os.listdir(root_path))

        for folder_name in folder_names:
            folder_path = os.path.join(root_path, folder_name)
            if os.path.isdir(folder_path):
                if 'NET' in folder_path:
                    D = 0
                    for a,_,c in os.walk(folder_path):
                        for f in c:
                            if D < (self.N+1) and 'TCP' in f :
                                D = D+1
                                image_path = os.path.join(a, f)
                                self.image_paths.append(image_path)
                                # Assign labels based on folder order
                                self.labels.append(self.label_mapping[folder_name])
                            
                else: 
                    image_paths = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path) if 'TCP' in image_name]
                    if len(image_paths) > (self.N+1):
                        image_paths = image_paths[1:self.N]
                        for image_path in image_paths:
                            self.image_paths.append(image_path)
                            self.labels.append(self.label_mapping[folder_name])
                    else: 
                        for image_path in image_paths:
                            self.image_paths.append(image_path)
                            self.labels.append(self.label_mapping[folder_name])
                

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(image_path)

        # Apply transformations if needed
        if self.transform:
            image = self.transform(image)

        return image, label


def VALval(model, dataloader, out_ss, flag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs, flag)
            # Thresholding
            predicted_classes = torch.argmax(outputs, dim=1)
            
            #  tensors arrays
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted_classes.cpu().numpy())

    # Calculate F1 score
    #f1 = f1_score(y_true, y_pred, average='macro') 
    #     f1 = calculate_f1_scoreY(y_true, y_pred) #
    metric = MulticlassF1Score( num_classes=out_ss)
    f1  = metric(torch.tensor(y_true), torch.tensor(y_pred) )
    cf = confusion_matrix(y_true, y_pred)
    
    
    return f1, cf


def print_confusion_matrix(conf_matrix, labels):
    index_to_label = {v: k for k, v in labels.items()}
    print("Confusion Matrix:")
    print(f"{'':>20}", end="")
    for label in labels:
        print(f"{index_to_label[label]:>20}", end="")
    print()
    for i, row in enumerate(conf_matrix):
        print(f"{index_to_label[labels[i]]:>20}", end="")
        for val in row:
            print(f"{val:>20}", end="")
        print()

def VALval_with_confusion_matrix(model, dataloader, label_mapping, flag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs, flag)
            # Thresholding
            predicted_classes = torch.argmax(outputs, dim=1)

            # Collect tensors arrays
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted_classes.cpu().numpy())

    # Calculate confusion matrix
    labels = list(label_mapping.values())
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

    # Optional: Print confusion matrix with class names
    print_confusion_matrix(conf_matrix, labels)

    return conf_matrix




class BalanceSetSynth(DatasetFolder):
    def __init__(self, root_path_real,root_path_fake,  transform=None, N=100, rate_fake=0.5):
        self.root_path_real = root_path_real
        self.root_path_fake = root_path_fake
        self.transform = transform
        self.N = N  # Number of examples to keep for each class 
        self.N_fake = int(round(N*rate_fake,0)) # Number of fake examples for class
        self.N_real = int(self.N - self.N_fake)
        self.image_paths = []
        self.labels = []
        # Get a list of folder names (classes)
        self.folder_names = sorted(os.listdir(root_path_real))
        
        
        self.label_mapping = {
            'hydra_ssh2': 0,
            'java_rmi': 1,
            'mirai': 2,
            'netbios_ssn2': 3,
            'replayAttacks': 4
            }
        
        self.get_list(self.N_real, self.root_path_real, 0)
        self.get_list( self.N_fake, self.root_path_fake, 1)
        
                    
    def get_list(self, Num,root_f , glag=0):
        for folder_name in self.folder_names:
            folder_path = os.path.join(root_f, folder_name)
            if os.path.isdir(folder_path):
                if glag == 0:
                    if 'NET' in folder_path:
                        D = 0
                        for a,_,c in os.walk(folder_path):
                            for f in c:
                                if D < (Num) and 'TCP' in f :
                                    D = D+1
                                    image_path = os.path.join(a, f)
                                    self.image_paths.append(image_path)
                                    # Assign labels based on folder order
                                    self.labels.append(self.label_mapping[folder_name])
                                
                    else: 
                        image_paths = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path) if 'TCP' in image_name]
                        if len(image_paths) > (Num+1):
                            image_paths = image_paths[1:Num]
                            for image_path in image_paths:
                                self.image_paths.append(image_path)
                                self.labels.append(self.label_mapping[folder_name])
                else:
                    image_paths = [os.path.join(folder_path, image_name) for image_name in os.listdir(folder_path)]
                    if len(image_paths) > (Num+1):
                        image_paths = image_paths[1:Num]
                        for image_path in image_paths:
                            self.image_paths.append(image_path)
                            self.labels.append(self.label_mapping[folder_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(image_path)

        # Apply transformations if needed
        if self.transform:
            image = self.transform(image)

        return image, label

