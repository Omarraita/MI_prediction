import numpy as np
from skimage.filters import frangi
from skimage.color import rgb2gray
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class ContrastiveLearningFrangiGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, frangi_transform, base_transform):
        self.frangi_transform = frangi_transform
        self.base_transform = base_transform

    def __call__(self, x):

        image = self.frangi_transform(x)
        img = (np.transpose(image.cpu().detach().numpy(), (1, 2, 0))*255)
        gray_img = rgb2gray(img)

        filtered_1 = frangi(gray_img,sigmas=range(2, 5, 3)) 
        filtered_2 = frangi(gray_img,sigmas=range(5, 8, 3)) 

        filtered_3 = frangi(gray_img,sigmas=range(8, 12, 3)) 
        #plt.imsave('./filtered_1.jpg', filtered_1, cmap='gray')
        #plt.imsave('./filtered_2.jpg', filtered_2, cmap='gray')
        #plt.imsave('./filtered_3.jpg', filtered_3, cmap='gray')
        filtered_1 = transforms.ToTensor()(filtered_1).float() 
        filtered_2 = transforms.ToTensor()(filtered_2).float() 
        filtered_3 = transforms.ToTensor()(filtered_3).float() 

        filtered = torch.cat([filtered_1, filtered_2, filtered_3], dim=0)

        return [self.base_transform(x), filtered]
