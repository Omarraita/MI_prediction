# Set Base Project Folder
base_folder = '/content/drive/MyDrive/Robotics project I/'
#base_folder = '/home/thanou/'
import sys
sys.path.append(base_folder)
from implementations import *
from utils import *
import sys
sys.path.append(base_folder)
import cv2
import os
import re
import numpy as np
import math
import random
from pathlib import Path
import imageio
from PIL import Image
from skimage import io, transform
from sklearn.model_selection import train_test_split


def delete_unpairs (folder_labelled, folder_raw) :
  '''
  This function checks the labeleld images and raw images to ensure that each 
  labelled image has a corresponding raw image. Any images that are not 
  paired will be removed.

  folder_labelled : path of folder containing the labelled images
  folder_raw : path of folder containing the raw images
  '''
  nameslab=[]
  namesraw=[]
  for filename2 in os.listdir(folder_raw) :
      namesraw.append(filename2)
  for filename in os.listdir(folder_labelled) :
      filename_new = re.sub(r'_[A-Za-z0-9]{2}', '', filename)
      filename_new = re.sub(r'[A-Za-z]{5}','', filename_new)
      nameslab.append(filename_new)
      if filename_new not in namesraw :
          os.remove(os.path.join(folder_labelled,filename))
  for filename2 in os.listdir(folder_raw) :
      if filename2 not in nameslab :
          os.remove(os.path.join(folder_raw,filename2)) 

def read_labels (img) :

  ''' Iterates over each labelled image in search of red or green pixels.
      Saves location of each detected colored dot. Detects the colored dots by 
      comparing the R and G color values to the other color channels. '''

  pxRed = []
  pxGreen = []
  nx, ny, c = img.shape
  label = 0

  # iterate over each pixel, checking if it is red or green
  for x in range (nx):
    for y in range (ny):
      pixel = img[x,y]

      if pixel[0]>200 :
        if pixel[0]>(pixel[1]+100):
          label=1
          pxRed.append((x, y))
          # when a colored pixel is detected, an area around it is blocked 
          # out to avoid detecting multiple pixels for a single dot
          img[x-15:x+15, y-15:y+15] = 0 

      if pixel[1]>200 :
        if pixel[1]>(pixel[0]+100):
          pxGreen.append((x, y))
          img[x-15:x+15, y-15:y+15] = 0

  return pxRed, pxGreen



def crop_patch (img, img_ann, pxRed, pxGreen, position) :

  ''' For input location, will crop largest possible square patch 
      up to 224x224 pixels containing only 1 dot and inside the outer
      borders of the image.'''

  nx, ny = img.shape[0], img.shape[1]
  x0 = position[0]
  y0 = position[1]
  red=pxRed.copy()
  green = pxGreen.copy()

  # determine label of patch based on if the center position is in the red or green list
  if position in pxRed :
    red.remove(position)
    label = 1
  else :
    green.remove(position)
    label = 0
  
  # combine all points without center point
  points = red + green

  # Calculating the distance between the center, all other points and the boundaries
  dist = [math.floor(math.sqrt((x0-x)**2 + (y0-y)**2)) for x, y in points]
  dist += [x0, y0, nx-x0, ny-y0, 224]

  # crop size of smallest measured distance
  size_patch = min(dist)

  patch = img[x0-size_patch//2:x0+size_patch//2, y0-size_patch//2:y0+size_patch//2,:]
  patch_annotated = img_ann[x0-size_patch//2:x0+size_patch//2, y0-size_patch//2:y0+size_patch//2,:]
  print(patch.shape)
    
  return patch, patch_annotated, label

def save_data_(save_path, X, Y, filenames, annotated_patches):

    """Save image (X) and label (Y) data in ``.npz`` format.    """

    file_ = Path(save_path).with_suffix('.npz')
    file_.parent.mkdir(parents=True,exist_ok=True)
    np.savez(str(save_path), X=np.asarray(X), Y=np.asarray(Y), filenames=np.asarray(filenames), annotated_patches=np.asarray(annotated_patches))

def create_dataset (folder_labelled, folder_raw, save_path) :

  ''' Takes folder location of both labelled and raw images
      and outputs a .npz file with patches around dots
      and labels.'''

  images = []
  annotated_patches = []
  filenames = []
  labels = []

  for f in os.listdir(folder_labelled) :
    # Keep the initial annotated image since the img_ref instance will be modified
    img_ann = cv2.imread(os.path.join(folder_labelled, f))
    img_ann = cv2.cvtColor(img_ann, cv2.COLOR_BGR2RGB)
    # remove prefix of labelled imgs to match raw imgs
    filename = re.sub(r'_[A-Za-z0-9]{2}', '', f)
    filename = re.sub(r'[A-Za-z]{5}','', filename)

    
    # load image and convert colors to RGB
    img_ref = cv2.imread(os.path.join(folder_labelled, f))

    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

    pxRed, pxGreen = read_labels(img_ref)

  
    image = io.imread(os.path.join(folder_raw, filename))

    #annotated_image = io.imread(os.path.join(folder_labelled, f))

    if pxRed :
      for position in pxRed :
        patch, annotated_patch, label = crop_patch(image, img_ann, pxRed, pxGreen, position) #added
        annotated_patches.append(annotated_patch) #added
        images.append(patch)
        labels.append(label)
        filenames.append(filename) #added
    if pxGreen :
      for position in pxGreen :
        patch, annotated_patch, label = crop_patch(image,img_ann, pxRed, pxGreen, position) #added
        annotated_patches.append(annotated_patch) #added
        images.append(patch)
        labels.append(label)
        filenames.append(filename) #added
  
  
  save_data_(save_path, images, labels, filenames, annotated_patches)


def augment_red(data_file, aug_factor):

  '''This function duplicates the patches containing a red dot in order 
  to balance the amount of green and red dots.'''

  X,Y = load_data(data_file)

  red_inds = np.where(Y == 1)
  for i in red_inds :
    image = X[i]
    label = Y[i]
    for j in range(aug_factor):
      X = np.append(X, image)
      Y = np.append(Y, label)
  # shuffling of the training set after augmentation of red data
  indices = np.arange(X.shape[0])
  np.random.shuffle(indices)

  filename = data_file[:-4] + '_augRed.npz'
  save_data(filename, X[indices], Y[indices])


# Labelled and raw image folders: (to be modified with your folders)


folder_labelled = base_folder + 'Raw_Data/Labelled/'
folder_raw = base_folder + 'Raw_Data/Raw/'

# File to save dataset patches

patches_file = base_folder + 'Datasets/dataset_patches.npz'

# Files to save train, validation and test patches

train_file = base_folder + 'Datasets/training_patches.npz'
val_file = base_folder + 'Datasets/validation_patches.npz'
test_file = base_folder + 'Datasets/test_patches.npz'
train_cv_file = base_folder + 'Datasets/training_patches_cv.npz' # train file specifically for cross validation

# check for and remove unpaired (label/raw) images
delete_unpairs(folder_labelled, folder_raw)

# create patches file (this cell takes 30-40 minutes)
create_dataset(folder_labelled, folder_raw, patches_file)

# split patches into test and train sets
training_test_split(patches_file, train_file, test_file, test_ratio=0.1, seed=1)

# save separate train set for cross validation:
# we do this in order to have more data in the cross validaton folds
X, Y = load_data(train_file)
save_data(train_cv_file, X, Y)

# split patches into train and validation sets
training_test_split(train_file, train_file, val_file, test_ratio=0.18, seed=1)

# Augment red points in all sets
'''
aug_factor = 3
augment_red(train_file, aug_factor)
augment_red(val_file, aug_factor)
augment_red(test_file, aug_factor)
augment_red(train_cv_file, aug_factor)
'''
