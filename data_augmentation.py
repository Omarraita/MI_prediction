import random
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import albumentations as A
import sys
from implementations import *
from utils import *


def data_augmentation(patches_file,trsf, base_folder, all_aug_factor, test_ratio, val_ratio):
  '''
  Augments the red patches in the patches_file so that the data becomes balanced and tops it by a whole data augmentation by all_aug_factor
  If all_aug_factor = 0, no augmentation will be performed.
  '''

  # Files to save train, validation and test patches
  train_file = base_folder + 'Datasets/training_patches.npz'
  val_file = base_folder + 'Datasets/validation_patches.npz'
  test_file = base_folder + 'Datasets/test_patches.npz'

  # split patches into test and train sets
  training_test_split2 (patches_file, train_file, test_file, test_ratio)

  # split patches into train and validation sets
  training_test_split2 (train_file, train_file, val_file, val_ratio)

  # Augment red points in all sets

  augment_red_2(train_file, trsf)
  augment_red_2(val_file, trsf)
  #augment_red_2(test_file, trsf, aug_factor)

  # New data augmentation from aug_red files
  train_file_ = base_folder + 'Datasets/training_patches_augRed.npz'
  val_file = base_folder + 'Datasets/validation_patches_augRed.npz'
  test_file = base_folder + 'Datasets/test_patches_augRed.npz'

  # Augment both red and green points by au_factor in all sets
  aug_factor = all_aug_factor
  augment_data(train_file_, trsf, aug_factor)
  augment_data(val_file, trsf, aug_factor)
  #augment_data(test_file, trsf, aug_factor)

def weighted_dataloader(trsf):
  '''
  Returns dataloader that takes into consideration the inequal data distribution between the two classes. No data augmentation is performed
  '''
  # Set base folder
  base_folder = '/content/drive/MyDrive/Robotics project I/'
  train_file = base_folder + 'Datasets/training_patches.npz'
  val_file = base_folder + 'Datasets/validation_patches.npz'
  test_file = base_folder + 'Datasets/test_patches.npz'

  ### Initialize data loader
  data_train = CardioDataset(train_file, trsf, to_tensor = True) # need to load data to get indices
  data_val = CardioDataset(val_file, trsf, to_tensor = True) # need to load data to get indices
  dataloader = []

  # Initialize the dataloader with sampler 
  Y = data_train.labels
  set_red = (Y==1) + 0
  class_red = sum(set_red)
  set_green = (Y==0) + 0
  class_green = sum(set_green)
  N = len(Y)
  print('Length of green patches class is: ',class_green)
  print('Length of red patches class is: ',class_red)
  print('# green patches / # red patches is: : ',class_green/class_red)
  weights = np.array([0.]*N)
  weights[Y==1] = 1/class_red
  weights[Y==0] = 1/class_green
  weights = torch.DoubleTensor(weights)
  sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))    
  mini_batch_size = 20
  dataloader = {'train': DataLoader(data_train,  mini_batch_size, sampler=sampler, num_workers=0), 'val': DataLoader(data_val,  mini_batch_size, shuffle=True, num_workers=0)}
  dataset_sizes = {'train': len(data_train), 'val': len(data_val)}
  print('Proportion of red_patches from the dataloader with a weighted sampler is: ')
  for sample in dataloader['train']:
    labels = sample['label']
    print((len(labels)-sum(labels))/len(labels)) 

  return dataloader 
