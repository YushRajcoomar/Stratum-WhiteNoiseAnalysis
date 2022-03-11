from typing import List
from cv2 import mean
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np
import torch



def grid_display(list_of_images : List,
                 list_of_titles : List,
                 no_of_columns : int,
                 figsize : tuple = (10, 10)):
    """displays images in a grid with a title

    Args:
        list_of_images (List[np.array]): list of numpy arrays
        list_of_titles (List[str]): list of plot headers in the same order as list of images
        no_of_columns (int): number of columns in grid
        figsize (tuple, optional): size of each image in grid. Defaults to (10,10).
    """    

    fig = plt.figure(figsize = figsize)
    column = 0

    for i in range(len(list_of_images)):
        column += 1
        #  check for end of column and create a new figure
        if column == no_of_columns+1:
            fig = plt.figure(figsize = figsize)
            column = 1

        fig.add_subplot(1, no_of_columns, column)

        plt.imshow(list_of_images[i])
        plt.axis('off')
        plt.title(list_of_titles[i])


    
    
def grid_histogram(data : np.array,
                   list_of_images : List,
                   rows : int,
                   columns : int,
                   figsize : tuple = (50, 50),
                   figheight : float = 15.0,
                   figwidth : float = 15.0,
                   hspace : float = 0.3):
    """plots the images in list of images and a histogram on the same row

    Args:
        data (np.array ): 1 dimension data to pass in histogram 
        list_of_images (tuple(nd.array)):  list of images is a list of (column - 1) dimension tuples
        rows (int): number of rows in grid display
        columns (int): number of columns in display, we assume the last column to be the histogram column
        figsize (tuple, float): tuple of ints to adjust height and width of displays. Defaults to (50,50).
        figheight (float): height of every histogram. Defaults to 15.
        figwidth (float): width of every histogram. Defaults to 15.
        hspace (float): horizontal space between subplots. Defaults to 0.3.
    """    
    # Initializing plot dimensions
    f, axarr = plt.subplots(rows, columns)
    f.set_figheight(figheight)
    f.set_figwidth(figwidth)
    
    n_classes = rows - 1
    
    for i,j in enumerate((list_of_images)):
        # Get different images from each tuple in the array
        prediction_arr = data[i].numpy()[0]
        noise_factor = j[0]
        injected_img = j[1]
        
        axarr[i, 0].imshow(noise_factor)
        axarr[i, 1].imshow(injected_img)
        axarr[i, 2].bar(range(n_classes), prediction_arr)

        axarr[i, 2].set_xticks(list(range(n_classes)))
        axarr[i, 2].set_yticks([0.5, 1])
        
    f.subplots_adjust(hspace)
    f.show()


def display_data(data : np.ndarray,
                 targets : np.ndarray,
                 class_names : List,
                 number_of_columns : int):
    """displays an image in each class in the dataset

    Args:
        data (ndarray[int]): data array
        targets (ndarray[int]): array of all classes
        class_names (List[str]): names of classes
        number_of_columns (int): number of columns
    """

    unique_classes = np.unique(targets).astype(int)
    idx_arr = []

    # Appends 1 image per class to array
    for i in unique_classes:
        idx = np.where(targets == unique_classes[i])
        idx_arr.append(idx[0][i])

    # Displays images with a title header
    list_of_images = [data[i] for i in idx_arr]
    list_of_titles = class_names
    grid_display(list_of_images, list_of_titles, no_of_columns = number_of_columns)     
        
def orig_v_reconstructed(img_kernels : List,
                         weights : np.array,
                         train : torch.tensor,
                         img_id : int):
    """displays the difference between original image and a reconstructed image using gabor kernels.

    Args:
        img_kernels (List[nd.array]): Image kernels used to reconstruct original image
        weights (np.array): regression weights from gabor regression
        train (torch.datasets): train pytorch dataset object
        img_id (int): index of image
    """    
    # Flattens img_kernels
    X = np.stack(img_kernels, -1)
    X = X.reshape(-1, X.shape[-1])

    train_shape = train.data.shape

    # Matrix Multiplication - weights @ X. Applies kernels on image,
    reconstructed_image = weights.dot(X.T)[img_id].reshape(train_shape[1],train_shape[2])
    original_image = train.data[img_id]

    grid_display(list_of_images = [reconstructed_image, original_image],
                      list_of_titles = ['Reconstructed','Original'],
                      no_of_columns = 2,
                      figsize=(5,5))


def loss_curve(train_loss : List,
               val_loss : List,
               title : str):
    """returns training and validation loss curves during neural network training

    Args:
        train_loss (List): train losses accumulated during training
        val_loss (List): validation losses accumulated during training
        title (str): Title of graph
    """    

    '''
    
    train_loss : List -> 
    val_loss : List -> 
    title : str -> t
    '''
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def accuracy_curve(train_accuracy : List,
                   val_accuracy : List,
                   title : str):
    '''
    returns training and validation accuracy curves during neural network training
    train_accuracy : List -> train accuracy accumulated during training
    val_loss : List -> accuracy losses accumulated during training
    val_accuracy : str -> title of graph
    '''
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def noise_maps(noise : np.array,
               n_samples : int):
    """displays n_samples of random noise

    Args:
        noise (np.ndarray): noise array
        n_samples (int): number of samples to be displayed
    """    
    # Collecting first n_samples
    noise_list = [noise[i] for i in range(n_samples)]
    noise_names = ["Noise Sample {}".format(i+1) for i in range(n_samples)]
    num_columns = n_samples//2

    grid_display(list_of_images = noise_list,
                      list_of_titles = noise_names,
                      no_of_columns = num_columns)

def bias_map(noise_data : dict):
    """displays the mean activation layers

    Args:
        noise_data (dict[int : np.array]): noise_data is a dict containing the frequency of predictions when noise is passed in the network.
                                           Each image is the mean activation layer per class.
    """        
    n_classes = len(noise_data.keys())

    mean_noise, titles = [], []

    for cls in range(n_classes):
        # Mean image per class
        a = torch.sqrt((noise_data[cls]**2).mean(0)).numpy()
        a = (a - a.min()) / (a.max() - a.min())

        mean_noise.append(a)
        titles.append(cls)

    grid_display(list_of_images= mean_noise,
                      list_of_titles= titles,
                      no_of_columns= n_classes//5 )

def confusion_matrix(y_pred : torch.Tensor,
                     y_test : torch.Tensor,
                     classes : List,
                     title : str,
                     normalize : bool = True):
    """Displays a confusion matrix for the classification task

    Args:
        y_pred (torch.Tensor): Predicted classes
        y_test (torch.Tensor): Ground truth
        classes (List): List containing all classes
        title (str): Plot header
        normalize (bool, optional): Displays an accuracy percentage if True, an integer if False. Defaults to True.
    """                     
    # Classification Metrics
    class_metrics = metrics.precision_recall_fscore_support(y_pred, y_test, average = 'macro')
    accuracy = (y_pred == y_test).sum()/len(y_test)
    metric_str = 'Accuracy: {}\nPrecision: {}\nRecall: {}\nF1score: {}'.format(
                                                                 np.round(accuracy,3),
                                                                 np.round(class_metrics[0],3),
                                                                 np.round(class_metrics[1],3),
                                                                 np.round(class_metrics[2],3))

    # Confusion Matrix
    cm = metrics.confusion_matrix(y_pred, y_test)
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] )

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Setting ticks
    ax.set(xticks=np.arange(cm.shape[1])-0.5,
       yticks=np.arange(cm.shape[0])-0.5,
       xticklabels=classes, yticklabels=classes,
       title=title,
       ylabel='Predicted label',
       xlabel='True label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.text(11, 5, metric_str, fontsize=12)

    # Annotations inside matrix
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



