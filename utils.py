import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from sklearn.metrics import confusion_matrix

from torchvision.utils import make_grid, save_image
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import os

def display_image_grid(images_filepaths, predicted_labels=(), cols=5, nr_of_samples = 10):

    ### Define a function to visualize images and their labels

    rows = nr_of_samples // cols
    images = np.load(images_filepaths, allow_pickle = True)
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    
    total_number = len(images.f.Y)
    randomList = random.sample(range(0, total_number), nr_of_samples)
    
    for i, j in enumerate(randomList):
        image = images.f.X[j]
        label = images.f.Y[j]
        if label == 0:
            true_label = "Non culprit"
        else:
            true_label = "Culprit"   
        #true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
        predicted_label = predicted_labels[j] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def visualize_augmentations(dataset, trsf, idx=0, samples=10, cols=5):
    ### Define a function to visualize augmeneted data

    
    dataset = copy.deepcopy(dataset)
    images = np.load(dataset, allow_pickle = True)
    #dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    image = images.f.X[idx]
    
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        transformed_image = trsf(image=image)['image']
        
        ax.ravel()[i].imshow(transformed_image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show() 

def imshow(inp, title=None):

    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, dataloader, device, class_names, num_images=6):

    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, sample in enumerate(dataloader['val']):
            
            inputs = sample['image'].to(device)
            
            labels = sample['label']
            
            labels = torch.max(labels, 1)[0]
            labels = labels.to(device)
            
            
            #if labels == 0:
             #   true_label = "Non culprit"
            #else:
             #   true_label = "Culprit"
            
            #inputs = inputs.to(device)
            #labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                #if class_names[preds[j]] == labels[j]:
                if preds[j] == labels[j]:
                    color = 'green'
                else: 
                    color = 'red'
                ax.set_title('predicted: {}'.format(class_names[preds[j]]), color = color)
                
                   
        #true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
            #predicted_label = predicted_labels[j] if predicted_labels else true_label
            #color = "green" if true_label == predicted_label else "red"
                imshow(inputs.cpu().data[j])
                #imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def performance_evaluation(prediction, true_label):

    error_vec = prediction-true_label

    false_pos = torch.count_nonzero(error_vec == 1)
    false_neg = torch.count_nonzero(error_vec == -1)

    positive = torch.count_nonzero(true_label == 1)
    negative = torch.count_nonzero(true_label == 0)

    return positive, negative, false_pos, false_neg

def sensitivity_performance(total_pos, total_neg, false_neg, false_pos):

    sensitivity = (total_pos - false_neg)/total_pos
    sensitivity = sensitivity.cpu().numpy()
    specificity = (total_neg - false_pos)/total_neg
    specificity = specificity.cpu().numpy()
    f1 = (total_pos-false_neg)/(total_pos-false_neg+0.5*false_pos+0.5*false_neg)
    f1 = f1.cpu().numpy()

    return sensitivity, specificity, f1

def conf_matrix(true_label, prediction):

    tn, fp, fn, tp = confusion_matrix(true_label, prediction).ravel()

    return tn, fp, fn, tp


def display_gradcam(model, model_name, base_folder, visualization_file, data_train, y_true, all_outputs, random_index, Frg):
    """
    Displays and saves gradcam visualization with the model's prediction on a random sample from data_train folder
    model: model to evaluate
    model_name (String): model's name for savefig
    data_train: data from which random sample is retrieved (not necessarily the training data)
    random_index (Int): Index of sample to select
    Frg (bool): Specifies wether the model is a Frangi model or not
    """
    
    architecture = model
    test_model = model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    normed_torch_img = data_train[random_index]['image'].unsqueeze(0).to(device)
    f = np.load(visualization_file, allow_pickle = True)
    _, _, _, annotated_patches = f['X'], f['Y'], f['filenames'], f['annotated_patches']
    annotated_patch = annotated_patches[random_index]

    if Frg:
        normed_torch_filtered_img = data_train[random_index]['filtered_1'].unsqueeze(0).to(device)
        normed_torch_img = torch.cat([normed_torch_img,normed_torch_filtered_img], dim=1).float() 

    # Set configs for heatmap
    configs = [dict(model_type='resnet', arch=architecture, layer_name='layer4')]

    for config in configs:
        config['arch'].to(device).eval()

    cams = [ [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)] for config in configs]

    # Test prediction
    
    images = []
    for gradcam, gradcam_pp in cams:
        
        mask, _ = gradcam(normed_torch_img)

        heatmap, result = visualize_cam(mask, normed_torch_img[:,0:3,:,:])

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, normed_torch_img[:,0:3,:,:])
      
        #images.extend([normed_torch_img[:,0:3,:,:].cpu(), heatmap, heatmap_pp, result, result_pp])
        images.extend([normed_torch_img[:,0:3,:,:].cpu(), result_pp])

    images[0] = images[0][0]

    grid_image = make_grid(images, nrow=5)
    class_names = ['Non culprit','Culprit']

    if not os.path.exists(base_folder+'Gradcam_images'):
        os.makedirs(base_folder+'Gradcam_images')

    for i, image in enumerate(images):
        fig = plt.figure()
        ax = plt.subplot()
        if all_outputs[random_index].int() == y_true[random_index].int():
            color = 'green'
        else: 
            color = 'red'
        ax.set_title(model_name+' predicted: {}'.format(class_names[all_outputs[random_index].int()]), color = color)
        imshow(image)

    #plt.title('predicted: {}'.format(class_names[all_outputs[random_index].int()]), color = color)
    plt.title('Patch index: '+ str(random_index), color = color)
    fig.savefig(base_folder+'Gradcam_images/'+model_name+'_patch_'+str(random_index)+'.png')

    plt.imshow(annotated_patch)
    plt.savefig(base_folder+'Gradcam_images/annotated_patch_'+str(random_index)+'.png')
      
  
def plot_pretrained_vs_scratch(ft_hist_train, scratch_hist_train,  ft_hist, scratch_hist, ft_sensitivity, scratch_sensitivity, ft_specificity, scratch_specificity, ft_f1_score, scratch_f1_score):
  
  fig, axs = plt.subplots(2, 3)

  # Training accuracy
  ohist = [h.cpu().numpy() for h in ft_hist_train]
  shist = [h.cpu().numpy() for h in scratch_hist_train]

  axs[0, 0].plot(range(1,num_epochs+1), ohist,label="Pretrained")
  axs[0, 0].plot(range(1,num_epochs+1), shist,label="Scratch")
  axs[0, 0].set_ylim((0,1.))
  axs[0, 0].set_xticks(np.arange(1, num_epochs+1, 2.0))
  axs[0, 0].legend()
  axs[0, 0].set_xlabel("Epochs")
  axs[0, 0].set_ylabel("Training Accuracy")

  # Validation accuracy
  ohist = [h.cpu().numpy() for h in ft_hist]
  shist = [h.cpu().numpy() for h in scratch_hist]

  axs[0, 1].plot(range(1,num_epochs+1), ohist,label="Pretrained")
  axs[0, 1].plot(range(1,num_epochs+1), shist,label="Scratch")
  axs[0, 1].set_ylim((0,1.))
  axs[0, 1].set_xticks(np.arange(1, num_epochs+1, 2.0))
  axs[0, 1].legend()
  axs[0, 1].set_xlabel("Epochs")
  axs[0, 1].set_ylabel("Validation Accuracy")

  # Sensitivity
  ft = 'ft_sensitivity'
  scratch = 'scratch_sensitivity'

  ohist = [h for h in eval(ft)]
  shist = [h for h in eval(scratch)]

  axs[1, 0].plot(range(1,num_epochs+1), ohist,label="Pretrained")
  axs[1, 0].plot(range(1,num_epochs+1), shist,label="Scratch")
  axs[1, 0].set_ylim((0,1.))
  axs[1, 0].set_xticks(np.arange(1, num_epochs+1, 2.0))
  axs[1, 0].legend()
  axs[1, 0].set_xlabel("Epochs")
  axs[1, 0].set_ylabel("Sensitivity")

  # Specificity
  ft = 'ft_specificity'
  scratch = 'scratch_specificity'

  ohist = [h for h in eval(ft)]
  shist = [h for h in eval(scratch)]

  axs[1, 1].plot(range(1,num_epochs+1), ohist,label="Pretrained")
  axs[1, 1].plot(range(1,num_epochs+1), shist,label="Scratch")
  axs[1, 1].set_ylim((0,1.))
  axs[1, 1].set_xticks(np.arange(1, num_epochs+1, 2.0))
  axs[1, 1].legend()
  axs[1, 1].set_xlabel("Epochs")
  axs[1, 1].set_ylabel("Specificity")

  fig.set_figheight(7)
  fig.set_figwidth(12)

  # F1_Score
  variable = 'f1_score'

  ft = 'ft_f1_score'
  scratch = 'scratch_f1_score'

  ohist = [h for h in eval(ft)]
  shist = [h for h in eval(scratch)]

  axs[0, 2].plot(range(1,num_epochs+1), ohist,label="Pretrained")
  axs[0, 2].plot(range(1,num_epochs+1), shist,label="Scratch")
  axs[0, 2].set_ylim((0,1.))
  axs[0, 2].set_xticks(np.arange(1, num_epochs+1, 2.0))
  axs[0, 2].legend()
  axs[0, 2].set_xlabel("Epochs")
  axs[0, 2].set_ylabel("F1_score")

  fig.set_figheight(7)
  fig.set_figwidth(12)
  #fig.suptitle('Training performance after data balancing and 1x data augmentation', fontsize=16)

  axs[-1, -1].axis('off')

  fig.tight_layout()