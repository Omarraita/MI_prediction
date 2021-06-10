import random
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.models as modelsT
from torch.utils.data import Dataset, DataLoader
import time
import copy
import os
from utils import *

from skimage.filters import frangi, hessian 
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#*******************************************************
#                   BASIC functions
#*******************************************************

def load_data(file_path): 
    
    """Load training data from file in ``.npz`` format."""
    f = np.load(file_path, allow_pickle=True)
    X, Y = f['X'], f['Y']
    Y=np.squeeze(Y)
    return (X,Y)

def load_data2(file_path): 
    
    """Load training data from file in ``.npz`` format, with the filenames and the annotated patches for visualization"""
    f = np.load(file_path, allow_pickle=True)
    X, Y, filenames, annotated_patches = f['X'], f['Y'], f['filenames'], f['annotated_patches']
    Y=np.squeeze(Y)
    return (X,Y, filenames, annotated_patches)
    
    
def save_data(save_path, X, Y):

    """Save image (X) and label (Y) data in ``.npz`` format.    """

    file_ = Path(save_path).with_suffix('.npz')
    file_.parent.mkdir(parents=True,exist_ok=True)
    np.savez(str(save_path), X=np.asarray(X), Y=np.asarray(Y))

def save_data2(save_path, X, Y, filename, annotated_patches):

    """Save image, label, filenames and annotated patches data in ``.npz`` format.    """

    file_ = Path(save_path).with_suffix('.npz')
    file_.parent.mkdir(parents=True,exist_ok=True)
    np.savez(str(save_path), X=np.asarray(X), Y=np.asarray(Y), filenames = filename, annotated_patches=np.asarray(annotated_patches))
    
    
    
def training_test_split (data_file, train_save_file, test_save_file, test_ratio, seed = None):

    ''' Function uses scikit-learn's train_test_split() to split data and saves resulting
      .npz files '''

    X,Y = load_data(data_file)
 
    #x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, random_state=seed, stratify = Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_ratio, random_state=seed)

    save_data(test_save_file, x_test, y_test)
    save_data(train_save_file, x_train, y_train)

def training_test_split2 (data_file, train_save_file, test_save_file, test_ratio, seed = None):

    ''' Function uses scikit-learn's train_test_split() to split data with the filenames and the annotated patches and saves resulting
      .npz files '''

    X,Y,filenames, annotated_patches = load_data2(data_file)
 
    x_train, x_test, y_train, y_test, filename_train, filename_test, annotated_patches_train, annotated_patches_test  = train_test_split(X, Y,filenames, annotated_patches, test_size=test_ratio, random_state=seed, stratify = Y)

    save_data2(test_save_file, x_test, y_test, filename_test, annotated_patches_test)
    save_data2(train_save_file, x_train, y_train, filename_train, annotated_patches_train)
    

def augment_red(data_file, aug_factor):

      '''This function duplicates the patches containing a red dot in order 
      to balance the amount of green and red dots.'''

      #X,Y, filenames = load_data2(data_file)
      X,Y = load_data(data_file)

      red_inds = np.where(Y == 1)
      for i in red_inds :
        image = X[i]
        label = Y[i]
       # fname = filenames[i]
        for j in range(aug_factor):
          X = np.append(X, image)
          Y = np.append(Y, label)
      # shuffling of the training set after augmentation of red data
      indices = np.arange(X.shape[0])
      np.random.shuffle(indices)

      filename = data_file[:-4] + '_augRed.npz'
      #save_data(filename, X[indices], Y[indices], fname)
      save_data(filename, X[indices], Y[indices])

def augment_red_2(data_file, trsf):
    
      '''Replicates red patches in order to balanced the data and applies morphological transformations trsf, on the replicated patches.'''
      
      X,Y, filenames, annotated_patches = load_data2(data_file)
      X = X.tolist()
      annotated_patches = annotated_patches.tolist()
      red_inds = np.where(Y == 1)
      red_inds = np.array(red_inds).flatten()
      
      aug_factor = np.sum(Y==0)/np.sum(Y==1)
      print(aug_factor)
      max_counter = int((aug_factor-1)*np.sum(Y==1))
      print('aug is', max_counter)
      counter = 1
      i=0
      
      while True:
        pos = red_inds[i]
        
        image = X[pos]
        label = Y[pos]
        filename = filenames[pos]
        annotated_patch = annotated_patches[pos]
        
        transformed = trsf(image=image)['image']
        X.append(transformed)
        annotated_patches.append(annotated_patch)
        Y = np.append(Y, label)
        filenames = np.append(filenames, filename)
        
        i += 1
        if(i==len(red_inds)):
            i = 0
        counter +=1
        
        if(counter== max_counter):
            break

      # shuffling of the training set after augmentation of red data
      X = np.array(X)
      annotated_patches = np.array(annotated_patches)
      indices = np.arange(X.shape[0])
      np.random.shuffle(indices)
      
      filename = data_file[:-4] + '_augRed.npz'

      save_data2(filename, X[indices], Y[indices], filenames[indices], annotated_patches[indices])

def augment_data(data_file, trsf, aug_factor):
    
      '''Replicates the whole data with an augmentation factor aug_factor. Applies transformations trsf,to the replicated patches'''
      
      X,Y, filenames, annotated_patches = load_data2(data_file)
      X = X.tolist()
      annotated_patches = annotated_patches.tolist()
      red_inds = np.arange(0,len(Y),1) # Select all the data
      red_inds = np.array(red_inds).flatten()
      
      for i in red_inds :
        image = X[i]
        label = Y[i]
        filename = filenames[i]
        annotated_patch = annotated_patches[i]
        
        for j in range(aug_factor):
            transformed = trsf(image=image)['image']
            X.append(transformed)
            annotated_patches.append(annotated_patch)
            Y = np.append(Y, label)
            filenames = np.append(filenames, filename)

      # shuffling 
      X = np.array(X)
      annotated_patches = np.array(annotated_patches)
      indices = np.arange(X.shape[0])
      np.random.shuffle(indices)

      save_data2(data_file, X[indices], Y[indices], filenames[indices], annotated_patches[indices])
      
#*******************************************************
#                   Dataset Class
#*******************************************************    

class CardioDataset(Dataset):
    
    """Dataset of images and Labels. If want to visualize the images, set 
       to_normalize = False"""

    def __init__(self, npz_file, transform=None, to_tensor=True, to_normalize = True):
        """
        Args:
            npz_file (string): Path to the npz file with patches and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.patches, self.labels = load_data(npz_file)
        self.transform = transform
        self.tensor = to_tensor
        self.norm = to_normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.patches[idx]
        image = image.astype('float32')
        label = np.array([self.labels[idx]])
        

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
          
                  
        if self.tensor==True : 
            trsf = transforms.ToTensor()
            image = trsf(image)/255      # torch totensor does not rescale float inputs to 0-1 scale

        # normalize settings based on Pytorch pre-trained model reqs
        if self.norm==True :         
            trsf2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = trsf2(image)
          
        label = torch.from_numpy(label)

        sample = {'image': image , 'label': label}   
        
        return sample
        
        
class CardioDatasetFrangi(Dataset):
    
    """Custom dataset class that renders the patch, its label and the Frangi-filtered patch """

    def __init__(self, npz_file, transform=None, to_tensor=True, to_normalize = True, reverse=False):
        """
        Args:
            npz_file (string): Path to the npz file with patches and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            reverse (bool): Specifies wether tha rendered frangi image is reversed = 1- frangi or not.
                            The reversed frangi highlights the vessels with black color instead of white.
        """
        self.patches, self.labels, self.filenames, self.annotated = load_data2(npz_file)
        self.transform = transform
        self.tensor = to_tensor
        self.norm = to_normalize
        self.reverse = reverse

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.patches[idx]
        image = image.astype('float32')
        annotated = self.annotated[idx]
        
        label = np.array([self.labels[idx]])
        img = image

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
                  
        if self.tensor==True : 
            trsf = transforms.ToTensor()
            image = trsf(image)
            #Create filtered image
            img = image.cpu().detach().numpy().transpose(1, 2, 0)
            gray_img = rgb2gray(img)
            filtered_1 = frangi(gray_img,sigmas=range(5, 10, 5)) 

            image = image/255      # torch totensor does not rescale float inputs to 0-1 scale

        # Normalization
        if self.norm==True :         
            trsf2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            trsf3 = transforms.Normalize(mean=[0.9849009416635326], std=[0.05]) # Normalize reversed frangi
            trsf4 = transforms.Normalize(mean=[0.01519], std=[0.05079]) # Normalize frangi
            image = trsf2(image)

            if(self.reverse):
              filtered_1 = trsf3(trsf(1-filtered_1)) # Take complementary, apply to tensor and normalize 
            else:
              filtered_1 = trsf4(trsf(filtered_1)) # Apply to tensor and normalize 

        label = torch.from_numpy(label)

        sample = {'image': image ,'label': label, 'filtered_1': filtered_1}   
        
        return sample

def weighted_dataloader(train_file, val_file, trsf):
  '''
  Returns dataloader that takes into consideration the inequal data distribution between the two classes. No data augmentation is performed
  '''
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
  weighted_prop_green = []
  for sample in dataloader['train']:
    labels = sample['label']
    prop = (len(labels)-sum(labels))/len(labels)
    weighted_prop_green.append(prop.item()) 

  return dataloader, np.array(weighted_prop_green)     
  
#*******************************************************
#                  Models Initializer
#******************************************************* 
    
def initialize_model(modelName, num_classes=2, pretrained_model = True):
    
  
    ''' List of all models we tested and their respective
        initializations. Here we modify the final output layer to 
        2 classes'''
    model = None
    dropout = True

    if modelName == 'dense121':
        model = modelsT.densenet121(pretrained=pretrained_model)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    elif modelName == 'dense161':
        model = modelsT.densenet161(pretrained=pretrained_model)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
  
    elif modelName == 'dense201':
        model = modelsT.densenet201(pretrained=pretrained_model)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    elif modelName == 'resnet18':
        model = modelsT.resnet18(pretrained=pretrained_model)
        num_ftrs = model.fc.in_features

        if dropout:
            model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))
            print(dropout)
        else:
    
            model.fc = nn.Linear(num_ftrs, num_classes)

    elif modelName == 'resnet101':
        model =modelsT.resnet101(pretrained=pretrained_model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif modelName == 'resnet152':
        model = modelsT.resnet152(pretrained=pretrained_model)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif modelName == 'vgg11_bn':
        model = modelsT.vgg11_bn(pretrained=pretrained_model)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
  
    elif modelName == 'vgg13_bn':
        model = modelsT.vgg13_bn(pretrained=pretrained_model)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif modelName == 'vgg19_bn':
        model = modelsT.vgg19_bn(pretrained=pretrained_model)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)

    else:
        raise NameError('Model name not included in our list.')

    return model


#*******************************************************
#       Regular Train, Test and Predict Functions
#*******************************************************

def test_model(model, test_file, trsf, mini_batch_size):
    '''
    Tests the model's performance by performing forward passes on mini_batches and computing different performance metrics.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    model.eval()
      
    error_count = 0
    total_pos = 0; total_neg = 0
    false_pos = 0; false_neg = 0

    test_data = CardioDataset(test_file, transform = trsf, to_tensor = True)
    testloader = DataLoader(test_data,  mini_batch_size, shuffle=False, num_workers=0)

    for j_batch, sample_batched in enumerate(testloader):
        test_input = sample_batched['image']
        test_target = sample_batched['label']
        test_target = torch.max(test_target, 1)[0]
        test_input = test_input.to(device)
        test_target =test_target.to(device)          
          
        prediction = model(test_input)
        prediction = prediction.argmax(axis = 1)

        # Keep track of all target labels
        total_pos = total_pos + torch.count_nonzero(test_target == 1)
        total_neg = total_neg + torch.count_nonzero(test_target == 0)

        error_vec = prediction-test_target

        # based on error vec value, determine wheter false positive or false negative
        false_pos = false_pos + torch.count_nonzero(error_vec == 1)
        false_neg = false_neg + torch.count_nonzero(error_vec ==-1)

        error_count = error_count + torch.sum((error_vec) != 0)
          
    acc_te = 1-error_count/len(test_data)
    acc_te = acc_te.cpu().numpy()
    
    sensitivity = (total_pos - false_neg)/total_pos
    sensitivity = sensitivity.cpu().numpy()
    specificity = (total_neg - false_pos)/total_neg
    specificity = specificity.cpu().numpy()
    f1 = (total_pos-false_neg)/(total_pos-false_neg+0.5*false_pos+0.5*false_neg)
    f1 = f1.cpu().numpy()
        
    return acc_te, specificity, sensitivity, f1, total_pos, total_neg


def train_model_v3(model, dataloader, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    """
    Trains and evaluates model by iterating over the whole data, in two phases: 'train' and 'val'
    """
    since = time.time()
    
    val_acc_history = []
    train_acc_history = []
    
    sensitivity = []; specificity = []; f1_score = []; loss_val = []; loss_train = [];
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            total_pos = 0; total_neg = 0; total_false_neg = 0; total_false_pos = 0

            # Iterate over data.
            for sample in dataloader[phase]:
                
                inputs = sample['image'].to(device)
                labels = sample['label']
                labels = torch.max(labels, 1)[0]
                labels = labels.to(device)
            
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                positive, negative, false_pos, false_neg = performance_evaluation(preds, labels)
                
                total_pos += positive
                total_neg += negative
                total_false_neg += false_neg
                total_false_pos += false_pos
                
            if phase == 'train':

                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':

                loss_train.append(epoch_loss)
            else:
                loss_val.append(epoch_loss)

            sen, spe, f1 = sensitivity_performance(total_pos, total_neg, total_false_neg, total_false_pos)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val':
                print('Sensitivity: ', "{:.2f}".format(sen), ' Specificity: ', "{:.2f}".format(spe),
                      ' F1 score: ', "{:.2f}".format(f1))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                specificity.append(spe)
                sensitivity.append(sen)
                f1_score.append(f1)
            else:
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, specificity, sensitivity, f1_score, loss_train, loss_val


def train_model_Frg(model, dataloader, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    """
    Trains and evaluates a model, taking into consideration 4 channels input data. 
    3 channels for the initial patch plus one channel for the frangi image.
    """
    since = time.time()
    
    val_acc_history = []
    train_acc_history = []
    
    sensitivity = []; specificity = []; f1_score = []; loss_val = []; loss_train = [];
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            total_pos = 0; total_neg = 0; total_false_neg = 0; total_false_pos = 0

            # Iterate over data.
            for sample in dataloader[phase]:
                
                inputs = sample['image'].to(device)
                
                filtered = sample['filtered_1'].to(device)
                labels = sample['label']
                labels = torch.max(labels, 1)[0]
                labels = labels.to(device)
            
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    filtered = torch.reshape(filtered, (inputs.size()[0], 1, 224, 224))
                    inputs = torch.cat([inputs,filtered], dim=1)

                    inputs = inputs.to(device)
                    inputs = inputs.float()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                positive, negative, false_pos, false_neg = performance_evaluation(preds, labels)
                
                total_pos += positive
                total_neg += negative
                total_false_neg += false_neg
                total_false_pos += false_pos
                
            if phase == 'train':

                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':

                loss_train.append(epoch_loss)
            else:
                loss_val.append(epoch_loss)

            sen, spe, f1 = sensitivity_performance(total_pos, total_neg, total_false_neg, total_false_pos)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val':
                print('Sensitivity: ', "{:.2f}".format(sen), ' Specificity: ', "{:.2f}".format(spe),
                      ' F1 score: ', "{:.2f}".format(f1))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                specificity.append(spe)
                sensitivity.append(sen)
                f1_score.append(f1)
            else:
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, specificity, sensitivity, f1_score, loss_train, loss_val   

def plot_performance(base_folder, model_name, num_epochs, hist_test, hist_train, specificity, sensitivity, f1_score, loss_train, loss_val, save=False):
    """
    Plots and saves the model's performances: train and validation losses, train and validation accuracies, specificity, sensitivity and f1_score.
    """
    if not os.path.exists(base_folder+'Results'):
      os.makedirs(base_folder+'Results')
    
    fig = plt.figure()
    plt.plot(loss_val, color = 'r', alpha= 0.7, label = 'Validation loss')
    plt.plot(loss_train, color = 'b', alpha= 0.7, label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.grid()
    plt.xticks(np.arange(1, num_epochs+1, 2.0))
    plt.legend()
    if(save):
        plt.savefig(base_folder+'Results/'+model_name+'_losses.png')
    
    fig = plt.figure()
    plt.plot(hist_test,color = 'r', alpha= 0.7, label = 'Validation accuracy')
    plt.plot(hist_train, color = 'b', alpha= 0.7, label = 'Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.xticks(np.arange(1, num_epochs+1, 2.0))
    plt.legend()
    if(save):
        plt.savefig(base_folder+'Results/'+model_name+'_accuracies.png')
    
    fig = plt.figure()
    plt.plot(specificity,color = 'b',linestyle='--', alpha= 0.7, label = 'Specificity')
    plt.plot(sensitivity, color = 'r',linestyle='--', alpha= 0.7, label = 'Sensitivity')
    plt.plot(f1_score, color = 'black',linestyle='--', alpha= 0.8, label = 'F1-score')
    plt.xlabel('Epochs')
    plt.grid()
    plt.xticks(np.arange(1, num_epochs+1, 2.0))
    plt.legend()
    if(save):
        plt.savefig(base_folder+'Results/'+model_name+'_performance_metrics.png')
        
        
def test_model_conf_mat(model, test_data, mini_batch_size, Frg):
    """
    Computes the predictions of the model, the evaluation metrics and the confusion matrix.
    Frg (bool): specifies wether the passes model is a 3-channels or 4-channels input to compute 
    the forward pass accordingly.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prediction_mat = []
    all_outputs = torch.tensor([])
    y_true = torch.tensor([], dtype=torch.long)
  
    model.eval()
    model.to(device)
    error_count = 0
    total_pos = 0; total_neg = 0
    false_pos = 0; false_neg = 0

    testloader = DataLoader(test_data,  mini_batch_size, shuffle=False, num_workers=0)

    for sample_batched in testloader:
      
        test_input = sample_batched['image'].to(device).cpu()
        test_target = sample_batched['label'].to(device).cpu()
        test_target = torch.max(test_target, 1)[0]
        test_input = test_input.to(device).float()
        if(Frg): # If frangi is used, concatenate the test input with its filtered image.
          filtered = sample_batched['filtered_1'].to(device)
          filtered = torch.reshape(filtered, (test_input.size()[0], 1, 224, 224))
          test_input = torch.cat([test_input,filtered], dim=1).float()

        test_target =test_target.to(device).cpu()          

        prediction = model(test_input)
        prediction = prediction.argmax(axis = 1).cpu()  

        cm = confusion_matrix(prediction.view(-1), test_target.view(-1))
        prediction_mat.append(cm)

        y_true = torch.cat((y_true, test_target), 0)

        all_outputs = torch.cat((all_outputs, prediction), 0)


        # Keep track of all target labels
        total_pos = total_pos + torch.count_nonzero(test_target == 1)
        total_neg = total_neg + torch.count_nonzero(test_target == 0)

        error_vec = prediction-test_target

        # based on error vec value, determine wheter false positive or false negative 
        false_pos = false_pos + torch.count_nonzero(error_vec == 1)
        false_neg = false_neg + torch.count_nonzero(error_vec == -1)

        error_count = error_count + torch.sum((error_vec) != 0)
          
    acc_te = 1-error_count/len(test_data)
    acc_te = acc_te.cpu().numpy()
  
    sensitivity = (total_pos - false_neg)/total_pos
    sensitivity = sensitivity.cpu().numpy()
    specificity = (total_neg - false_pos)/total_neg
    specificity = specificity.cpu().numpy()
    f1 = (total_pos-false_neg)/(total_pos-false_neg+0.5*false_pos+0.5*false_neg)
    f1 = f1.cpu().numpy()

        
    return acc_te, specificity, sensitivity, f1, total_pos, total_neg, prediction_mat, y_true, all_outputs 

def test_model_ten_runs(test_file, base_folder, trsf, model, model_name, frangi=False, reversed=False, save=False):
  """
  Performs ten tests with different seeds, computes the mean and the variance of each performance metric over the runs.
  Generates performance plots, confusion matrix and saves the results.
  """
  acc = []
  spe = []
  sens = []
  f1_ = []

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model.to(device)
  for i in range(10):
    torch.manual_seed(i)

    model.eval()
    # Define dataset
    if(frangi):
      testing_data = CardioDatasetFrangi(test_file, trsf, to_tensor = True, reverse=reversed) 
    else:
      testing_data = CardioDataset(test_file, trsf, to_tensor = True) 

    acc_te, specificity, sensitivity, f1, total_pos, total_neg, prediction_mat, y_true, all_outputs = test_model_conf_mat(model, testing_data, 10, frangi)
    acc.append(acc_te)
    spe.append(specificity)
    sens.append(sensitivity)
    f1_.append(f1)
  

  cm = confusion_matrix(y_true.numpy(), all_outputs.numpy())

  disp = ConfusionMatrixDisplay(cm, display_labels=['Non culprit', 'Culprit'])
  disp.plot()
  plt.title(model_name)
  if not os.path.exists(base_folder+'test_results'):
    os.makedirs(base_folder+'test_results')
  if(save):
    plt.savefig(base_folder+'test_results/cm_'+model_name+'.png')

  data = [acc, spe, sens, f1_]
  labels = ['Accuracy','Specificity', 'Sensitivity', 'F1']
  fig7, ax7 = plt.subplots()
  ax7.set_title(model_name+ ' performance over 10 runs with different seeds')
  ax7.boxplot(data, labels=labels)
  plt.grid(axis = 'y')
  plt.tight_layout()
  if(save):
    plt.savefig(base_folder+'test_results/'+model_name+'.png')
    
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
#*******************************************************
#                 Cross Validation Functions
#*******************************************************

def build_k_indices(y, k_fold, seed):
    """
    y: output data
    k_fold: number of k-folds
    seed: seed for reproducibility
    build k indices for k-fold."""
 
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)    


def test_cv(model, test_data, mini_batch_size):
    
    ''' Modified from regular test function to accomodate subsettiing from k_folds indices. 
        Evaluate model on validation/test data. Outputs sensitivity, specificity, F1-score 
        and accuracy.'''

    model.eval()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    error_count = 0
    total_pos = 0; total_neg = 0
    false_pos = 0; false_neg = 0

    testloader = DataLoader(test_data,  mini_batch_size, shuffle=False, num_workers=0)

    for j_batch, sample_batched in enumerate(testloader):
        
        test_input = sample_batched['image']
        test_target = sample_batched['label']
        test_target = torch.max(test_target, 1)[0]
        test_input = test_input.to(device)
        test_target =test_target.to(device)          
          
        prediction = model(test_input)
        prediction = prediction.argmax(axis = 1)
    
        # Keep track of all target labels
        total_pos = total_pos + torch.count_nonzero(test_target == 1)
        total_neg = total_neg + torch.count_nonzero(test_target == 0)

        error_vec = prediction-test_target

        # based on error vec value, determine wheter false positive or false negative 
        false_pos = false_pos + torch.count_nonzero(error_vec == 1)
        false_neg = false_neg + torch.count_nonzero(error_vec == -1)

        error_count = error_count + torch.sum((error_vec) != 0)
          
    acc_te = 1-error_count/len(test_data)
    acc_te = acc_te.cpu().numpy()
  
    sensitivity = (total_pos - false_neg)/total_pos
    sensitivity = sensitivity.cpu().numpy()
    specificity = (total_neg - false_pos)/total_neg
    specificity = specificity.cpu().numpy()
    f1 = (total_pos-false_neg)/(total_pos-false_neg+0.5*false_pos+0.5*false_neg)
    f1 = f1.cpu().numpy()

    return acc_te, specificity, sensitivity, f1, total_pos, total_neg

def cross_validation_initialization(train_file, trsf, y, x, k_fold, num_epochs, modelName, lrates, mini_batch_set, weight_decay_set):
    """
    k_fold: rounds of training/validation
    lrates: list of learning rates to evaluate
    weight_decay_set: list of weight decays
    Returns  the mean of the model's performances over the folds (rounds)
    """
    seed = 12
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    # define lists to store the accuracy of training data and test data
    accuracies_tr = [[] for i in range(len(weight_decay_set))] # Stores average accuracy for each learning rate, for each batch_size
    accuracies_te = [[] for i in range(len(weight_decay_set))]
    specificities = [[] for i in range(len(weight_decay_set))]
    sensitivities = [[] for i in range(len(weight_decay_set))]
    f1_measures = [[] for i in range(len(weight_decay_set))]
    best_accuracy = 0
    
    for b in range(len(weight_decay_set)):

      # cross validation
      for i, lr in enumerate(lrates):
    
          acc_tr_tmp = []
          acc_te_tmp = []
          spe_tmp = []
          sens_tmp = []
          f1_tmp = []

          for k in range(k_fold):
              print('############################')
              print('Weight decay', weight_decay_set[b])
              print('Progression: wd #', b+1)
              print('Learning rate', lr)
              print('Progression: lr #', i+1)
              print('Fold', k)
              print('############################')
              model, acc_test, acc_train, specificity, sensitivity, f1_score = cross_validation(train_file, trsf, k_indices, k, lr, 20, weight_decay_set[b], num_epochs, modelName)

              acc_test_epochs = [float(h.cpu().numpy()) for h in acc_test]
              acc_te_tmp.append(np.mean(acc_test_epochs)) # Stores validation accuracy for each k, for a given lr, for a given

              acc_train_epochs = [float(h.cpu().numpy()) for h in acc_train]
              acc_tr_tmp.append(np.mean(acc_train_epochs)) # Stores training accuracy for each k, for a given lr

              spe_epochs = [float(h) for h in specificity]
              spe_tmp.append(np.mean(spe_epochs)) # Stores specificity for each k, for a given lr

              sens_epochs = [float(h) for h in sensitivity]
              sens_tmp.append(np.mean(sens_epochs)) # Stores sensitivity for each k, for a given lr

              f1_epochs = [float(h) for h in f1_score]
              f1_tmp.append(np.mean(f1_epochs)) # Stores f1 for each k, for a given lr
          if(np.mean(acc_te_tmp)>best_accuracy):
              best_accuracy = np.mean(acc_te_tmp)
              best_model = model
          accuracies_tr[b].append(np.mean(acc_tr_tmp)) # Mean over k_folds, for each lr
          accuracies_te[b].append(np.mean(acc_te_tmp)) # Mean over k_folds, for each lr
          specificities[b].append(np.mean(spe_tmp)) # Mean over k_folds, for each lr
          sensitivities[b].append(np.mean(sens_tmp)) # Mean over k_folds, for each lr
          f1_measures[b].append(np.mean(f1_tmp)) # Mean over k_folds, for each lr

    return best_model, accuracies_tr, accuracies_te, specificities, sensitivities, f1_measures

def cross_validation(train_file, trsf, k_indices, k, lr,mini_batch_size,weight_decay, num_epochs, modelName):
    """
    k_indices: subset of the input data used for training 
    k: the fold number
    returns the model's performance over the run corresponding to fold k and parameters lr, weight_decay
    """
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = np.reshape(tr_indice,(1,np.product(tr_indice.shape))).ravel()

    #data = CardioDataset(train_file, transform = trsf)
    data = CardioDatasetFrangi(train_file, trsf, to_tensor = True, reverse=False) 
    # subset data with k_fold inds
    train_data = torch.utils.data.Subset(data, tr_indice)
    test_data = torch.utils.data.Subset(data, te_indice)
    dataloader = []
    dataloader = {'train': DataLoader(train_data,  mini_batch_size, shuffle=True, num_workers=0), 'val': DataLoader(test_data,  mini_batch_size, shuffle=True, num_workers=0)}
    dataset_sizes = {'train': len(train_data), 'val': len(test_data)}

    pretrained = True
    num_classes = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_frangi = initialize_model(modelName, num_classes, pretrained)
    model_frangi = model_frangi.to(device)
    model_frangi.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_frangi = model_frangi.to(device) 

    scratch_optimizer = optim.SGD(model_frangi.parameters(), lr=lr,weight_decay=weight_decay, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(scratch_optimizer, step_size=7, gamma=0.1)
    scratch_criterion = nn.CrossEntropyLoss()
    model_frangi, acc_test, acc_train, specificity, sensitivity, f1_score, loss_train, loss_val = train_model_Frg(model_frangi, dataloader, dataset_sizes,  scratch_criterion, scratch_optimizer, exp_lr_scheduler,
                      num_epochs)
    
    return model_frangi, acc_test, acc_train, specificity, sensitivity, f1_score
    
    
def cross_validation_visualization(lambds, f1):
    """
    lambds: vector of tested lambdas used as an x axis
    f1: vector of all f1 measures
    """
    plt.plot(lambds, f1, marker=".", color='b')
    plt.xlabel("learning rates")
    plt.ylabel("F1 measure")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

    
def log_performance(lrates, weight_decay_set, metric_name, metrics, best_model, testing_data):

  metrics_names = {0:'validation accuracy', 1:'f1 measure', 2:'specifictiy', 3:'sensitivity'}

  idx = np.where(np.array(list(metrics_names.values()))==metric_name)[0].item()

  metric = metrics[idx]
  heatmap = np.array(metric)
  c = np.unravel_index(heatmap.argmax(), heatmap.shape)

  if not os.path.exists(base_folder+'CV_Results_frangi'):
      os.makedirs(base_folder+ 'CV_Results_frangi')
    
  f= open(base_folder+"CV_Results_frangi/test_results_"+metrics_names[idx] +".txt","w+")
  f.write("Best "+metrics_names[idx] +" %f\r\n" % heatmap[c])

  f.write("The corresponding tuple learning rate - weight decay is :  %f %s \r\n" % (lrates[c[1]],weight_decay_set[c[0]]))
  acc_test = np.array(metrics[0])[c]
  f.write("The corresponding testing accuracy is :  %f \r\n" %acc_test )
  f1 = np.array(metrics[1])[c]
  f.write("The corresponding F1_measure is :  %f \r\n" %f1 )
  spe = np.array(metrics[2])[c]
  sens = np.array(metrics[3])[c]
  f.write("The corresponding specificity - sensitivity is : %s %i \r\n" %(spe,sens))

  '''
  acc_te, specificity, sensitivity, f1, total_pos, total_neg = test_cv(best_model, testing_data, 20)
  f.write('Test set results are : \r\n' )
  f.write('Accuracy : %f\r\n' % acc_te)
  f.write('Specificity : %f\r\n' % specificity)
  f.write('Sensitivity : %f\r\n' % sensitivity)
  f.write('F1_measure : %f\r\n' % f1)
  f.close()
  '''

  # Heatmaps
  Z = np.array(metric)
  df = pd.DataFrame(data=np.around(Z,decimals=2), index = np.around(weight_decay_set,decimals=4), columns=np.around(lrates,decimals=4))
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax = sns.heatmap(df, annot = True, cbar_kws={'label': metrics_names[idx]})
  plt.xlabel("Learning rate")
  plt.ylabel("Weight decay") 
  plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
  plt.savefig(base_folder+'CV_Results_frangi/cv_'+metrics_names[idx] +'_heatmap')

  # 3D plots
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  
  X, Y = np.meshgrid(lrates, weight_decay_set)
  surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                        linewidth=0, antialiased=False)
  
  ax.set_xlabel('learning rate')
  ax.set_ylabel('weight decay')
  ax.set_zlabel(metrics_names[idx])
    
  plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
  plt.savefig(base_folder+'CV_Results_frangi/cv_'+metrics_names[idx] +'_2D_view1')
  ax.view_init(30, 120)

  plt.savefig(base_folder+'CV_Results_frangi/cv_'+metrics_names[idx] +'_2D_view2')
  ax.view_init(30, 60)

  plt.savefig(base_folder+'CV_Results_frangi/cv_'+metrics_names[idx] +'_2D_view3')