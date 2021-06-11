# MI_prediction

# Requirements

In order to run the notebooks, it is necessary to install the following packages:

pip install albumentations==0.4.6 (for the image transformations)

pip install thop (for SimSiam training)

pip install pytorch-gradcam (for the gradcam visualization)

# Data preparation

- Clone this repository into your base_folder. To run the data augmentation, download and extract Raw_Data containing raw and labelled angiography images from https://drive.switch.ch/index.php/s/367fkbeytfy24d8/authenticate (requires a password). 
We expect the directory structure to be the following:
base_folder/Raw_Data/  
  Raw/  
  Labelled/ 
  
 - The augmented data that was used to generate the results can be found in https://drive.switch.ch/index.php/s/Rh9UrhnUmVLjsFn (requires a password). If loaded manually, the expected     directory for the augmented data is:
 base_folder/Datasets/
   dataset_patches.npz
   training_patches_augRed.npz
   validation_patches_augRed.npz
   test_patches.npz

# Models 

Link to selected Models https://drive.switch.ch/index.php/s/VsSCiQ5uY3Leikr

Link to the model that was trained with a different task (stenosis prediction), on artificial data https://drive.switch.ch/index.php/s/GLZaagFUdPobYSv

# Notebooks

- Training.ipynb : This is the main file from which you can run the main experiment:

Data augmentation
Train different models: {0:'model_scratch', 1:'model_pretrained', 2:'model_frangi', 3:'model_frangi_reversed', 4:'model_simsiam', 5:'model_simsiam_frangi', 6:'model_art_data'}

Load and test the trained models.

Compare the predictions between the different models.

Display the Gradcam visualizations.

Display features with TSNE.

- Verifications.ipynb : 

Displays the patches extracted from the original images to perform sanity checks. 

Verifies and data coming from the weighted dataloader

Contains the display_patient_views(): Visualization function that displays the different patches coming from the a given patient with the corresponding label and prediction.

- Frangi-Net.ipynb: Contains a Frangi-Net implementation attempt. Might be useful for a future work.

# Python files
- label_detection_patch_creation_.py: Runnable file that detects the labels and crops the patches from the Raw Data.
- cross_validation_lr_wd_frangi.py: Runnable file to run a cross validation on both parameters learning rate and weight decay.
- implementations.py: contains all the implemented functions for this project. The main parts are: Data preparation, Dataset Classes, Models Initializer, Train and test functions, Cross validation functions.
- utils.py:  contains util functions such as visualization functions.

# SimSiam
- Contains the adapted PyTorch implementation of SimSiam (https://github.com/leftthomas/SimSiam) to the cardio dataset. Uses some functions taken from SimCLR implementation (https://github.com/sthalles/SimCLR). 

# Results Folders
- Results: Validation results for all models.
- CV_Results: Cross validation results for the regular resnet18 model.
- CV_Results_frangi: Cross validation results for the frangi model.
- Gradcam_images: Heatmap visualizations for different models.
- test_results: Test results for all models.

# Main results
- Performance of regular resnet18: 

![Regular_model](https://github.com/Omarraita/MI_prediction/blob/main/Results/model_pretrained_performance_metrics.png)

- Performance of the Frangi model: 

![Frangi_performance](https://github.com/Omarraita/MI_prediction/blob/main/Results/model_frangi_performance_metrics.png)

- Performance of SimSiam model:

![SimSiam_performance](https://github.com/Omarraita/MI_prediction/blob/main/Results/model_scratch_performance_metrics.png)

- Cross_validation on Frangi model:

![cross_val_3D](https://github.com/Omarraita/MI_prediction/blob/main/CV_Results_frangi/cv_f1%20measure_2D_view3.png)

![cross_val_heatmap](https://github.com/Omarraita/MI_prediction/blob/main/CV_Results_frangi/cv_f1%20measure_heatmap.png)
