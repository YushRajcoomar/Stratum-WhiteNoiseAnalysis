from ast import Num
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
        
    def train_model(self,loader,epochs,loss_function, optimizer, verbose=True):
        '''
        Performs forward and backward propagation to train neural network
        loader : torch.DataLoader -> dictionary of train and test data loaders
        epochs : int -> number of epochs to train neural network over
        loss_function : torch.nn -> loss function used for optimization
        optimizer -> torch.optim -> optimization algorithm for loss minimization
        '''

        '''
        assert types
        '''
        
        assert (len(loader.keys())) == 2, "loader must contain 2 dataloader objects"
        
        self.loader = loader
        self.loss_function = loss_function
        self.train_metrics = {'train_loss':[],'train_accuracy':[],
                              'val_loss':[], 'val_accuracy':[],
                              'epoch':0}
        self.train()
        train_loader = loader['train']

        for epoch in range(epochs):
            self.train_metrics['epoch'] = epoch+1
            
            for batch_idx, (data,target) in enumerate(train_loader):
                
                b_x = Variable(data).to(self.device) # batch x
                b_y = Variable(target).to(self.device)  # batch y
                
                #forward propagation
                optimizer.zero_grad()
                y_pred = self.forward(b_x)[0]
                loss = loss_function(y_pred,b_y)
                
                #back propagation
                loss.backward()
                optimizer.step()
                
            self.train_metrics['train_loss'].append(float(loss.cpu().data))
            
            self.train_eval()
            self.test_eval()
            self.train_logger(verbose)


    def train_logger(self,verbose):
        '''
        train_logger prints training metrics after each epoch

        verbose : bool -> verbose = True prints training metrics at the end of each epoch.
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
        self.eval()
        train_loader = self.loader['train']
        # Train prediction
        train_correct,train_total = 0,0
        for img,target in train_loader:
            
            img = img.to(self.device)
            
            with torch.no_grad():
                y_pred = self.forward(img)[0].max(1)[1]
                train_correct += (y_pred.cpu().numpy() == target.cpu().numpy()).sum()
                train_total += len(target)
        train_accuracy = train_correct/train_total
        self.train_metrics['train_accuracy'].append(train_accuracy)

    def test_eval(self):
        self.eval()
        test_loader = self.loader['test']

        # Test Prediction
        test_correct,test_total = 0,0
        test_loss = []
        for img,target in test_loader:
            with torch.no_grad():
                img = img.to(self.device)
                y_pred = self.forward(img)[0]
                argmax_y_pred = y_pred.max(1)[1]
                test_correct += (argmax_y_pred.cpu().numpy() == target.cpu().numpy()).sum()
                test_loss.append(self.loss_function(y_pred.cpu(),target))
                test_total += len(target)
        test_accuracy = test_correct/test_total
        test_loss_mean = sum(test_loss)/len(test_loss)
        self.train_metrics['val_accuracy'].append(test_accuracy)
        self.train_metrics['val_loss'].append(test_loss_mean)


    def prediction_transform(self,mean_noise, data):
        '''
        applies noise transformation to data and returns predicted class
        mean_noise : list -> mean noise per class
        test_data : Tensor[Tensor] -> data to be transformed
        '''
        
        n_classes =  len(mean_noise)
        avg_vec = np.stack(mean_noise).reshape(n_classes, -1)
        templates = torch.from_numpy(avg_vec).permute(1, 0) # swap dimensions
        templates = templates / torch.norm(templates, p=2, dim=0) # normalize 
        
        x = data.view(data.shape[0], -1)
        x = x / torch.cat([torch.norm(x, p=2, dim=1).view(-1,1)] * x.shape[1], 1) # Normalization
        pred = torch.mm(x, templates).max(1)[1] #Transform
        
        return pred
    
    def metrics_report(self,y_pred,y_test):
        """Returns a table of classification metrics such as Accuracy, Precision, Recall, F1-score for each class

        Args:
            y_pred (Tensor[int]): Predicted classes
            y_test (Tensor[int]): Ground Truth

        Returns:
            classification_rep (pd.DataFrame): DataFrame with classification metrics for each class
        """        '''
        '''
        cm =  confusion_matrix(y_pred,y_test)
        accuracy_per_class = {i:0 for i in range(len(cm))}
        class_report = classification_report(y_pred,y_test,output_dict=True)
        for i in range(len(cm)):
            accuracy_per_class[i]= cm[i,i]/cm[i,:].sum()
            class_report[str(i)].update({'accuracy':accuracy_per_class[i]})
        classification_rep = pd.DataFrame(class_report)
        return classification_rep.iloc[:,:len(cm)].sort_index()
    
    def classification_metrics(self,y_pred,y_test):
        """aggregates classification metrics for the entire model

        Args:
            y_pred (Tensor[int]): Predicted classes
            y_test (Tensor[int]): Ground Truth

        Returns:
            classification_metrics_dict (dict[str:float]): classification metrics for model
        """        '''
        '''
        self.classification_metrics_dict={'precision':0,'recall':0,'f1score':0,'support':0}
        metrics_arr =  precision_recall_fscore_support(y_pred,y_test,average="macro")
        accuracy = float((y_pred == y_test).sum()/len(y_test))
        for i,j in enumerate(self.classification_metrics_dict.keys()):
            self.classification_metrics_dict[j] = metrics_arr[i]
        self.classification_metrics_dict.update({'accuracy':accuracy})
        self.classification_metrics_dict = dict(sorted(self.classification_metrics_dict.items()))
        return self.classification_metrics_dict

        

        
        
        
        