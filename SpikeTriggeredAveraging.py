import torch
import numpy as np
import math
from tqdm import tqdm
import sys

class SpikeTriggeredAveraging:
    def __init__(self,model, pixel):
        self.model = model
        self.pixel = pixel
        self.rf_info = None
        pass

    def layer_dimensions_out(self,
                             layer_cur : List,
                             layer_prev : List):
        """layer_dimensions_out calculates the sizes of parameters of a given input matrix after convolution & pooling layers

        Args:
            layer_cur (List): list of float in order [kernel size, stride, padding, num channels]
            layer_prev (List): list of float in order [start, jump, rf, num_features ]

        Returns:
            List: List of float in order [start, jump, rf, num_features ]
        """        

        start_cur , jump_cur = layer_prev[0] , layer_prev[1]
        rf_cur, num_features_cur = layer_prev[2] , layer_prev[3]

        kernel,stride,padding = layer_cur[0], layer_cur[1], layer_cur[2]

        # Calculates the dimensions of output channel after kernels and padding has been applied
        num_features_out = math.floor((num_features_cur - kernel + 2*padding) / stride) + 1

        padding_mid = (num_features_out - 1)*stride - num_features_cur + kernel
        padding_left = math.floor(padding_mid/2)

        jump_out = jump_cur * stride
        rf_out = rf_cur + (kernel - 1)*jump_cur
        start_out = start_cur + ((kernel-1)/2 - padding_left)*jump_cur

        return [start_out, jump_out, rf_out,num_features_out , layer_cur[3]]

    def generate_rf_info(self,
                         layer_names : List,
                         layer_list : List):
        """generates a dictionary with the required dimensions after each layer to check for padding

        Args:
            layer_names (list[str]): name of layers as defined in model object
            layer_list (list[int]): list containing sizes of each layer in the order
                                    [filter size, stride, padding, output_dimension]
        
        Returns:
            rf_info (dict[str:List]): returns dimensions after passing each layer.
        """        
        self.rf_info = {i:None for i in layer_names}

        # lists are in the order [start, jump_out, rf_out, num_features, num_channels ]
        first_layer = [0.5,1, 1, self.pixel, 0]

        # Computes output dimensions after passing through first layer
        traversal_dim = self.layer_dimensions_out(layer_list[0],first_layer)
        self.rf_info[layer_names[0]] = traversal_dim
        
        # Computes output dimensions after passing each layer
        for i in range(1,len(layer_list)):
            print(traversal_dim)
            traversal_dim = self.layer_dimensions_out(layer_list[i],traversal_dim)
            self.rf_info[layer_names[i]] = traversal_dim
        return self.rf_info

    def padded_rf(self,
                  layer_name : str,
                  image : np.ndarray,
                  i : int,
                  j : int):
        """Adds a padding layer around the image if necessary

        Args:
            layer_name (str): name of layer in neural network
            image (np.ndarray): noise array
            i (int): starting position on x axis
            j (int): starting position on y axis

        Returns:
            cur_rf (ndarray): returns a padded image
        """        

        if self.rf_info is None:
            print("Please initialize rf_info")

        p_rf = self.rf_info[layer_name][2]

        # Center position from starting point
        center_i = self.rf_info[layer_name][0] + i * self.rf_info[layer_name][1]
        center_j = self.rf_info[layer_name][0] + j * self.rf_info[layer_name][1]
        
        # Movement from center position in 4 directions.
        left = int(center_i - p_rf / 2)
        right = int(center_i + p_rf / 2)
        up = int(center_j - p_rf / 2)
        bottom = int(center_j + p_rf / 2)

        cur_rf = image[max(up, 0): min(bottom, self.pixel), max(left, 0): min(right, self.pixel)]

        if left < 0: # pad left
            
            tmp = torch.zeros(cur_rf.shape[1], cur_rf.shape[2] - left)
            tmp[:, -left:] , cur_rf = cur_rf, tmp
            
        if up < 0: # pad up
            
            tmp = torch.zeros(cur_rf.shape[1] - up, cur_rf.shape[2])
            tmp[-up:, :], cur_rf = cur_rf, tmp
            
        if right > self.pixel: # pad right
            
            tmp = torch.zeros(cur_rf.shape[1], cur_rf.shape[2] + (right - self.pixel))
            tmp[:, : -(right-self.pixel)], cur_rf = cur_rf, tmp
            
        if bottom > self.pixel: # pad bottom
            
            tmp = torch.zeros(cur_rf.shape[1] + (bottom - self.pixel), cur_rf.shape[2])
            tmp[: -(bottom-self.pixel), :], cur_rf = cur_rf, tmp
            
        return cur_rf

    def get_features_from_cnn(self,
                              conv_layers : int,
                              train_data : np.ndarray,
                              batch_size : int,
                              pos_x : int,
                              pos_y : int):
        """get_features_from_cnn passes the training data and captures the output of every forward pass for each convolutional layer

        Args:
            conv_layers (int): number of convolutional layers in network
            train_data (ndarray): train_data passed in CNN
            batch_size (int): size of batches to iterate over
            pos_x (int): starting x-axis position on image
            pos_y (int): starting y-axis position on image

        Returns:
            conv_activations (dict[str:Tensor]): returns a tuple of activation layers for every forward pass in each convolutional layer
        """        
        model = self.model
        device = model.device

        conv_activations = {'conv{}_activation'.format(i+1) : torch.zeros(0) for i in range(conv_layers)}

        iteration = len(train_data) // batch_size
        with tqdm(total=iteration, file=sys.stdout) as pbar:
            for i in range(iteration):

                # Pass batch through model
                input_ = train_data[i * batch_size : (i+1) * batch_size]
                input_ = input_.type_as(torch.FloatTensor()).to(device)

                # Prediction, conv layer 1, conv layer 2,...
                with torch.no_grad():
                    model_outputs = model(input_[:,None,...])

                predictions = model_outputs[0]
                convolutional_layers = model_outputs[1:]

                # adds the convolutional layers to the dictionary
                for idx,key in enumerate(conv_activations.keys()):
                    conv_activations[key] = torch.cat((conv_activations[key],convolutional_layers[idx][:, :, pos_x, pos_y].cpu()))
                pbar.update(1)

        return conv_activations