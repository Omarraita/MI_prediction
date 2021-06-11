# MI_prediction

# Requirements
In order to run the notebooks, it is necessary to install the following packages:

pip install albumentations==0.4.6 (for the image transformations)
pip install thop (for SimSiam training)
pip install pytorch-gradcam (for the gradcam visualization)

# Data preparation
Clone this repository into your base_folder. Download and extract Raw_Data containing raw and labelled angiographic images from https://drive.switch.ch/index.php/s/367fkbeytfy24d8/authenticate (requires a password). 
We expect the directory structure to be the following:
base_folder/Raw_Data/  
  Raw/  
  Labelled/ 
  
 The augmented data that was used to generate the results can be found in https://drive.switch.ch/index.php/s/Rh9UrhnUmVLjsFn (requires a password). If loaded manually, the expected     directory for the augmented data is:
 base_folder/Datasets/
   dataset_patches.npz
   training_patches_augRed.npz
   validation_patches_augRed.npz
   test_patches.npz

# Models 
Link to selected Models https://drive.switch.ch/index.php/s/VsSCiQ5uY3Leikr
Link to the model that was trained with a different task (stenosis prediction), on artificial data https://drive.switch.ch/index.php/s/GLZaagFUdPobYSv

# Notebooks
Data_Verification.ipynb : This notebook is used to display the patches extracted from the original images. It allows to verify that the data has been augmented/transformed properly.  

Training.ipynb : This is the main file as it containing different sections. Allows to perform a data augmentation, parameters selection, training, displaying results and testing.

# Python files


# Main Results

Performance of the Resent18 model: 
![Resnet18_performance](Results/Resnet18_performance.png)

Cross_validation:
![cross_val](Results/cross_validation.png)
