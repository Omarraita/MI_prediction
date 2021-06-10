import random
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from albumentations.pytorch import ToTensorV2
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
cudnn.benchmark = True
import os
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
#from focal_loss.focal_loss import FocalLoss
import sys
from implementations import *
from utils import *

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import pandas as pd


#base_folder = '/home/raita/'
base_folder = '/content/drive/MyDrive/Robotics project I/'
sys.path.append(base_folder)


# Initialize tranforms composition
trsf = A.Compose([A.MedianBlur(blur_limit=5, p=1),  
                  A.Rotate (limit=45, interpolation=1, border_mode=4, p=0.75),
                  A.ShiftScaleRotate (shift_limit=0.1, scale_limit=0, rotate_limit=0, interpolation=1, border_mode=4, p=0.75),
                  A.Resize(224, 224)
                  ])  

### Define network parameters
modelName = 'resnet18'

### Set paths to access data
patches_file = base_folder + 'Datasets/dataset_patches.npz'
train_file = base_folder + 'Datasets/training_patches.npz'
val_file = base_folder + 'Datasets/validation_patches.npz'
test_file = base_folder + 'Datasets/test_patches.npz'
model_folder = base_folder + '/Results/'

#Load data
file = np.load(train_file, allow_pickle = True) #train_file
Y = file.f.Y
X = file.f.X

num_epochs = 20
mini_batch_set = np.array([5, 15]) 
lrates = np.linspace(0.0005, 0.005, 10)
weight_decay_set = np.linspace(0.0005, 0.2, 10)
k_fold = 5

# Run cross validation
best_model, accuracies_tr, accuracies_te, specificities, sensitivities, f1_measures = cross_validation_initialization(train_file, trsf, Y, X, k_fold, num_epochs, modelName, lrates, mini_batch_set, weight_decay_set)

# Test the best model
testing_data = CardioDataset(test_file, trsf, to_tensor = True) # need to load data to get indices

metrics = [accuracies_te, f1_measures, specificities, sensitivities]
metrics_names = {0:'validation accuracy', 1:'f1 measure', 2:'specifictiy', 3:'sensitivity'}

# Log performance for validation accuracy
log_performance(lrates, weight_decay_set, metrics_names[0], metrics, best_model, testing_data)

# Log performance for f1
log_performance(lrates, weight_decay_set, metrics_names[1], metrics, best_model, testing_data)

# Log performance for specificity
log_performance(lrates, weight_decay_set, metrics_names[2], metrics, best_model, testing_data)

# Log performance for sensitivity
log_performance(lrates, weight_decay_set, metrics_names[3], metrics, best_model, testing_data)