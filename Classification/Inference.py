import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from model import Network
import configparser
from ut import VALval_with_confusion_matrix

# CONF
config = configparser.ConfigParser()
config.read('/home/chiara/DataAUG/Classification/con.ini')
train_folder = config['AAA']['train_folder']
out_ss = len(os.listdir(train_folder))
ss = int(config['AAA']['SS'])
drop = float(config['AAA']['DROP'])

# Define the label mapping
label_mapping = {
    'Backdoor_attack': 0,
    'DDoS_HTTP_Flood_Attacks': 1,
    'NET': 2,
    'Port_Scanning_attack': 3,
    'Ransomware_attack': 4,
    'SQL_injection_attack': 5,
    'Uploading_attack': 6,
    'Vulnerability_scanner_attack': 7
}

# Create reverse mapping for easy lookup
index_to_label = {v: k for k, v in label_mapping.items()}



Models = os.listdir('/home/chiara/DataAUG/Classification/Model/')

path_tmp = os.join('/home/chiara/DataAUG/Classification/Model/',Models[0])

tmp_model = Network(1, out_ss, ss =ss, p = drop, up =0 )
tmp_model.load_state_dict(torch.load(path_tmp))




# Example usage (assuming 'model', 'dataloader', 'out_ss', and 'flag' are predefined)
# conf_matrix = VALval_with_confusion_matrix(model, dataloader, label_mapping, 0)
# print(conf_matrix)