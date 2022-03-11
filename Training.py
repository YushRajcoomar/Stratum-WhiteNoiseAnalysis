from ast import Num
from typing import List
import torch
import torch.utils.data.dataloader as dataloader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix,classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import sys



class Training:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_metrics = {'train_loss':[],'train_accuracy':[],
                              'val_loss':[], 'val_accuracy':[],
                              'epoch':0}
        
    def train_model(self,
                    loader : torch.DataLoader,
                    epochs : int ,
                    loss_function : torch.nn.LossFunction,
                    optimizer : torch.optim,
                    verbose : bool = True):
        """Performs forward and backward propagation to train neural network

        Args:
            loader (torch.DataLoader): dictionary of train and test data loaders
            epochs (int): number of epochs to train neural network over
            loss_function (torch.nn.LossFunction): loss function used for optimization
            optimizer (torch.optim): optimization algorithm for loss minimization
            verbose (bool, optional): If True, prints training metrics after each epoch. Defaults to True.
        """        
        
        assert (len(loader.keys())) == 2, "loader must contain 2 dataloader objects"
        
        # Initialization
        self.loader = loader
        self.loss_function = loss_function
        self.train()
        train_loader = loader['train']

        # Train network over epochs
        for epoch in range(epochs):

            self.train_metrics['epoch'] = epoch+1

            for data,target in train_loader:
                
                b_x = Variable(data).to(self.device) # batch x
                b_y = Variable(target).to(self.device)  # batch y
                
                #forward propagation
                optimizer.zero_grad()
                y_pred = self.forward(b_x)[0]
                loss = loss_function(y_pred,b_y)
                
                #back propagation
                loss.backward()
                optimizer.step()
            
            # Prints training and validation metrics after each epoch
            self.train_metrics['train_loss'].append(float(loss.cpu().data))
            self.train_eval()
            self.test_eval()
            self.train_logger(verbose)


    def train_logger(self,
                    verbose : bool):
        '''
        train_logger prints training metrics after each epoch

        verbose : (bool, optional) -> verbose = True prints training metrics at the end of each epoch.
        '''

        if verbose:
              train_metrics = self.train_metrics
              print('Epoch: {} Train Loss: {:.5f} Train Accuracy: {:.5f}\
                    Test Loss: {:.5f} Test Accuracy: {:.5f}\n'.format(train_metrics['epoch'],
                                                                      train_metrics['train_loss'][-1],
                                                                      train_metrics['train_accuracy'][-1],
                                                                      train_metrics['val_loss'][-1],
                                                                      train_metrics['val_accuracy'][-1]))

    def train_eval(self):
        """ Evaluates the model's training accuracy at it's current state during model training. 
            Records accuracy in self.train_metrics
        """ 

        self.eval()
        train_loader = self.loader['train']
        train_correct,train_total = 0,0

        # Passes every batch in data loader to model
        for img,target in train_loader:

            img = img.to(self.device) 
            # Predicts batch input
            with torch.no_grad():

                y_pred = self.forward(img)[0].max(1)[1]
                train_correct += (y_pred.cpu().numpy() == target.cpu().numpy()).sum()
                train_total += len(target)

        # Calculate accuracy
        train_accuracy = train_correct/train_total
        self.train_metrics['train_accuracy'].append(train_accuracy)

    def test_eval(self):
        """ Evaluates the model's validation accuracy at it's current state during model training. 
            Records accuracy and loss in self.train_metrics
        """ 

        self.eval()
        test_loader = self.loader['test']
        test_correct,test_total = 0,0
        test_loss = []

        # Passes every batch in data loader to model
        for img,target in test_loader:

            with torch.no_grad():
                # Pass image to model and get predictions
                img = img.to(self.device)
                y_pred = self.forward(img)[0]
                argmax_y_pred = y_pred.max(1)[1]

                # Get validation loss
                test_correct += (argmax_y_pred.cpu().numpy() == target.cpu().numpy()).sum()
                test_loss.append(self.loss_function(y_pred.cpu(), target))
                test_total += len(target)

        # Get validation accuracy
        test_accuracy = test_correct/test_total
        test_loss_mean = sum(test_loss)/len(test_loss)

        self.train_metrics['val_accuracy'].append(test_accuracy)
        self.train_metrics['val_loss'].append(test_loss_mean)


    def prediction_transform(self,
                            mean_noise : List,
                            data : torch.Tensor):
        """applies noise transformation to data and returns predicted class

        Args:
            mean_noise (list): mean noise per class
            data (Tensor[Tensor]): data to be transformed

        Returns:
            transformed_prediction (Tensor[Tensor]) predicted classes of transformed data
        """        

        n_classes =  len(mean_noise)
        avg_vec = np.stack(mean_noise).reshape(n_classes, -1)

        # swaps dimensions and normalizes data
        templates = torch.from_numpy(avg_vec).permute(1, 0)
        templates = templates / torch.norm(templates, p=2, dim=0)

        data = data.view(data.shape[0], -1) # Flattens
        data = data / torch.cat([torch.norm(data, p=2, dim=1).view(-1,1)] * data.shape[1], 1) # Normalization
        transformed_data = torch.mm(data, templates).max(1)[1] # Transform
        transformed_prediction = transformed_data.max(1)[1] # Argmax
        
        return transformed_prediction
    
    def metrics_report(self,
                       y_pred : torch.Tensor,
                       y_test : torch.Tensor):
        """Returns a table of classification metrics such as Accuracy, Precision, Recall, F1-score for each class

        Args:
            y_pred (Tensor[int]): Predicted classes
            y_test (Tensor[int]): Ground Truth

        Returns:
            classification_rep (pd.DataFrame): DataFrame with classification metrics for each class
        """

        # Gets precision, recall and F1 score 
        class_report = classification_report(y_pred,y_test,output_dict=True)

        # Gets confusion matrix to extract class-wise accuracy
        cm =  confusion_matrix(y_pred,y_test)
        accuracy_per_class = {i:0 for i in range(len(cm))}

        for i in range(len(cm)):
            accuracy_per_class[i]= cm[i,i]/cm[i,:].sum()
            class_report[str(i)].update({'accuracy':accuracy_per_class[i]})

        classification_rep = pd.DataFrame(class_report)
        return classification_rep.iloc[:,:len(cm)].sort_index()
    
    def classification_metrics(self,
                                y_pred : torch.Tensor,
                                y_test : torch.Tensor):
        """aggregates classification metrics for the entire model

        Args:
            y_pred (Tensor[int]): Predicted classes
            y_test (Tensor[int]): Ground Truth

        Returns:
            classification_metrics_dict (dict[str:float]): classification metrics for model
        """

        self.classification_metrics_dict={'precision':0,'recall':0,'f1score':0,'support':0}

        # Gets classification metrics for each class
        metrics_arr =  precision_recall_fscore_support(y_pred,y_test,average="macro")
        accuracy = float((y_pred == y_test).sum()/len(y_test))

        # Updates dictionary with metrics
        for i,j in enumerate(self.classification_metrics_dict.keys()):
            self.classification_metrics_dict[j] = metrics_arr[i]
        self.classification_metrics_dict.update({'accuracy':accuracy})
        self.classification_metrics_dict = dict(sorted(self.classification_metrics_dict.items()))

        return self.classification_metrics_dict

        

        
        
        
        