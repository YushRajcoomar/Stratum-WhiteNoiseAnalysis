import cv2
import numpy as np
from tqdm import tqdm
import sys
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from Visualization import *
import torch

class Gabor():
    def __init__(self,
                pixel : int):
        self.pixel = pixel
        
    def normalize(self,
                  arr : List):
        '''
        arr : List -> List of data to normalize
        '''
        arr = (arr - arr.min())/(arr.max()-arr.min())
        return arr
        

    
    def generate_gabor_kernels(self,
                               kernel_size : List,
                               sigma_factor : float,
                               n_theta : List,
                               lambd_factor : float,
                               gamma : float,
                               n_psi : List,
                               ktype : cv2.type):
        """_summary_

        Args:
            kernel_size (List): list of kernel sizes
            sigma_factor (float): division factor for standard deviation of envelope
            n_theta (List): List of orientation of the normal to the parallel stripes of a Gabor function
            lambd_factor (float): division factor for wavelength of sinusoidal term
            gamma (float): Spatial aspect ratio
            n_psi (List): List of phase offset
            ktype (cv2.type): Type of filter coefficients
        """        

        self.gabor_kwargs = {'kernel_size':kernel_size,'sigma_factor':sigma_factor,
                             'n_theta':n_theta, 'lambda_factor':lambd_factor, 'n_psi':n_psi, 'ktype':ktype}
        self.kernels = {'gabor':[]}
        
        # number of kernels to cover length
        repetition_ct = [int(np.ceil(self.pixel/i)) for i in kernel_size]
        
        for i in range(len(repetition_ct)):
            for _theta in n_theta:
                for _psi in n_psi:
                    for rep in range(repetition_ct[i]**2): # Squared since images are p x p
                        k = kernel_size[i]
                        g_kernel = cv2.getGaborKernel((k, k), k/sigma_factor, _theta, k/lambd_factor,gamma, _psi, ktype) 
                        g_kernel = self.normalize(g_kernel)
                        self.kernels['gabor'].append(g_kernel)
        
        print("{} Gabor kernels have been generated".format(len(self.kernels['gabor'])))
    
    def generate_image_kernels(self):
        """From the gabor kernels generated above, image kernels are created to fit the image size.
        """        

        self.kernels.update({'image':[]})
        gabor_args = self.gabor_kwargs
        
        
        # number of kernels to cover length
        repetition_ct = [int(np.ceil(self.pixel/i)) for i in gabor_args['kernel_size']]
        counter = 0

        with tqdm(total=len(repetition_ct), file=sys.stdout) as pbar:
            for idx in range(len(repetition_ct)):

                cur_ct = repetition_ct[idx]
                k = gabor_args['kernel_size'][idx]
                iter_len = len(gabor_args['n_theta']) * len(gabor_args['n_psi'])

                for _ in range(iter_len):
                    for i in range(cur_ct):
                        for j in range(cur_ct):
                            tmp = np.zeros((self.pixel, self.pixel))

                            length = min(k*(i+1), self.pixel) - k*i
                            width = min(k*(j+1), self.pixel) - k*j

                            tmp[k*i: k*i + length, k*j: k*j + width] = self.kernels['gabor'][counter][:length, :width]
                            self.kernels['image'].append(tmp)
                            counter+=1

                pbar.update(1)
        print("{} image kernels generated".format(len(self.kernels['image'])))
                            
    def generate_noise(self,
                       train_data : torch.DataLoader,
                       reg_alpha : float,
                       pca_dim : int,
                       n_noise : int,
                       save_as : str,
                       save_bool : bool):
        """generate_noise creates and saves n gabor noise samples in cache folder

        Args:
            train_data (torch.DataLoader): train dataloader
            reg_alpha (float): regularization parameter for ridge regression
            pca_dim (int): number of dimensions to preserve in PCA
            n_noise (int): number of noise samples to generate
            save_as (str): name of noise file to be saved as. .npy extension not required
            save_bool (bool): if True, noise file is saved at save_as location. 
                              Note that for large n_noise, the file gets very heavy.
        """        

        # Ridge Regression to find weights
        img_kernel = self.kernels['image']
        X = np.stack(img_kernel, -1)
        X = X.reshape(-1, X.shape[-1])

        y = train_data
        y = y.permute(1, 2, 0).numpy() #swaps dimension axes
        y = y.reshape(-1, y.shape[-1])

        l2_reg = Ridge(alpha = reg_alpha)
        l2_reg.fit(X, y)

        self.reg_weights = l2_reg.coef_
        print("Regression weights have shape {}\n".format(self.reg_weights .shape))
        
        # PCA to reduce dimensionality of weights
        pca = PCA(n_components=pca_dim)
        principalComponents = pca.fit_transform(self.reg_weights)
        
        print("The explained variance with {} features is {:.3f}\n".format(pca_dim,pca.explained_variance_ratio_.sum()))
        print("The transformed data has shape {}\n".format(principalComponents.shape))
        
        # Generates random noise of size n_noise
        noise_wts = np.random.rand(n_noise, pca_dim)

        with tqdm(total=pca_dim, file=sys.stdout) as pbar:
            for i in range(pca_dim):
                cur_max = principalComponents[:, i].max()
                cur_min = principalComponents[:, i].min()
                cur_wts = self.normalize(noise_wts[:, i])
                noise_wts[:, i] = cur_wts * (cur_max - cur_min) + cur_min
    
                pbar.update(1)
        X = np.stack(img_kernel)
        X = X.reshape(X.shape[0], -1)
        noise_gabor_wts = noise_wts.dot(pca.components_)
        
        noise = noise_gabor_wts.dot(X).reshape(n_noise, self.pixel, self.pixel)
        for i in range(n_noise):
            noise[i] = self.normalize(noise[i])
        
        print("Generated noise samples have shape {}".format(noise.shape))
        if save_bool:
            save_str = 'cache/' + save_as + '.npy'
            np.save(save_str,noise)
            print("Gabor noise samples saved at {}".format(save_str))
    
    def noise_frequency(self,
                        model : torch.nn,
                        noise : np.ndarray,
                        batch_size : int,
                        n_classes : int):
        """noise_frequency creates a dictionary with classes as keys and predicted noise belonging to the class as the value

        Args:
            model (torch.nn): _description_
            noise (np.ndarray): generated noise samples
            batch_size (int): batches of noise to collect from noise data
            n_classes (int): number of classes

        Returns:
            noise_data (dict[int:Tensor]): dictionary contains the class number as key and Tensors as values.
                                           The tensor arrays are later used to find mean activation layers
        """                        


        noise_data = {i:[] for i in range(n_classes)}
        noise_length = noise.shape[0]

        model.eval()

        with tqdm(total=noise_length//batch_size,file=sys.stdout) as pbar:
            for i in range(noise_length//batch_size):

                # Preparing data to pass through model
                data = torch.from_numpy(noise[i * batch_size: (i+1) * batch_size]).float()
                data = data[:,None,...].to(model.device)
                
                with torch.no_grad():
                    y_pred = model.forward(data)[0].max(1)[1]

                # If prediction equals the group number, it is added to the list for that group.
                for group in range(n_classes):
                    noise_data[group].append(data[y_pred==group].cpu())
            pbar.update(1)

        for group in range(n_classes):
            noise_data[group] = torch.cat(noise_data[group])
            print(group, ':', noise_data[group].shape[0])

        return noise_data
            