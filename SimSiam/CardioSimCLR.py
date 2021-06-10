from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from gaussian_blur import GaussianBlur
from view_generator import ContrastiveLearningViewGenerator
import numpy as np 
import torch
from PIL import Image
from contrastive_frangi_gen import ContrastiveLearningFrangiGenerator

class CardioSimCLRDataset(Dataset):
    
    """Dataset of images. Returns unlabelled dataset for contrastive learning"""
    def __init__(self, npz_file, transform=None, to_tensor=False, to_normalize = False):
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
        img = Image.fromarray(image)
        label = np.array([self.labels[idx]])
        
        if self.transform is not None:
            image = self.transform(img)
            
        if self.tensor==True : 
            trsf = transforms.ToTensor()
            image = trsf(image)/255
            
        if self.norm==True :         
            trsf2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = trsf2(image)
            
        label = torch.from_numpy(label)
        
        sample = (image, label) 
    
        return image, label

def get_simclr_transform(size, use_frangi, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.2 * s, 0.2 * s, 0.2 * s, 0.1 * s)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if(use_frangi):
          
          data_transforms = transforms.Compose([transforms.Resize(size=size),
                                                GaussianBlur(kernel_size=int(0.1 * size)),
                                                transforms.ToTensor(),
                                                normalize,
                                              ])
        else:
          data_transforms = transforms.Compose([transforms.Resize(size=size),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomApply([color_jitter], p=0.1),
                                      transforms.RandomGrayscale(p=0.4),
                                      GaussianBlur(kernel_size=int(0.1 * size)),
                                      transforms.ToTensor(),
                                      normalize,
                                    ])
                                            
        return data_transforms

def get_frangi_transform(size, s=1):
        """ Only apply Gaussian blurring before computing frangi. """
        data_transforms = transforms.Compose([transforms.Resize(size=size),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor(),
                                             ])
        return data_transforms

def load_data(file_path): 
    """Load training data from file in ``.npz`` format."""
    f = np.load(file_path, allow_pickle=True)
    X, Y = f['X'], f['Y']
    Y=np.squeeze(Y)
    return (X,Y)

def get_cardio_smclr(train_file, use_frangi):

    if(use_frangi): #ContrastiveLearningFrangiGenerator
      print('Using frangi ...')
      train_data = CardioSimCLRDataset(train_file, transform = ContrastiveLearningFrangiGenerator(get_frangi_transform(96),get_simclr_transform(96,use_frangi)))
    
    else: #ContrastiveLearningViewGenerator
      print('Not using frangi ...')
      train_data = CardioSimCLRDataset(train_file, transform = ContrastiveLearningViewGenerator(get_simclr_transform(96, use_frangi), 2))
    
    return train_data
