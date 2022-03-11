import torch
import numpy as np
from skimage import exposure


class Denoise:
    def __init__(self,model):
        self.model = model
        pass

    def make_mean_activation_layer(self,
                                   size : int,
                                   batch_size : int,
                                   n_classes : int,
                                   input_size : int):
        """generates random noise of length size and returns a dictionary of frquencies of predicted classes 
            and a mean noise image from each of the predicted classes

        Args:
            size (int):  size of generated noise
            batch_size (int): size of batches to iterate over
            n_classes (int): number of target classes
            input_size (int): size of input image

        Returns:
            prediction_frequency (dict[int:Tensor]): Frequency of predicted classes
            mean_noise (list[Tensor]): list of mean activation images for each class at position index
        """        

        device = self.model.device
        
        mean_noise = torch.zeros(n_classes, input_size , input_size)
        noise = torch.rand(size, 1, input_size, input_size).to(device)
                
        output_tensor,prediction_tensor = torch.zeros(0).to(device),torch.zeros(0).to(device)

        # Pass noise through model
        for i in range(0, size,batch_size):
            prediction = self.model.forward(noise[i:i+batch_size])[0]
            output_layer, prediction = prediction.data.max(1)
            output_tensor = torch.cat((output_tensor, output_layer))
            prediction_tensor = torch.cat((prediction_tensor, prediction))
        
        # keys are classes and values are the frequency of predicted noise.
        prediction_frequency = {i:(prediction_tensor == i).sum()  for i in range(n_classes)}
        
        for i in range(n_classes):
            mean_prediction = torch.mean(noise[prediction_tensor==i] , dim=0)
            mean_noise[i] = mean_prediction.reshape(input_size,input_size)
        
        #set NAN to 0
        mean_noise[mean_noise != mean_noise] = 0 
        
        return prediction_frequency,mean_noise



    def noise_injection(self,
                        mean_noise : List,
                        weights : float,
                        test_data : torch.Tensor,
                        image_idx : int):
        """given a series of mean noise patterns, noise of degree weights is applied to the test_img

        Args:
            mean_noise (list[Tensor]): mean_noise pattern for class at index
            weights (float): proportion of noise applied to image 
            test_data (Tensor[Tensor]): Tensor of Tensors containing test dataset
            image_idx (int): index of selected image
        Returns:
            prediction_arr (list[Tensor]): prediction of noise samples when passed in data
            list_of_hist_images (list[ndarray]): collection of histogram for predictions
        """        


        device = self.model.device
        
        # Adjusts size and type of data to pass in the model
        test_data = test_data.type(torch.FloatTensor)
        test_data = (test_data - test_data.min())/(test_data.max() - test_data.min())

        prediction_arr, list_of_hist_images = [], []

        # fixed noise
        fixed_noise = torch.ones_like(torch.squeeze(test_data[image_idx,...],0))

        # base image
        base_image = torch.squeeze(test_data[image_idx,...],0)

        # base prediction
        test_image = test_data[image_idx][None,None,...]
        pixel = test_image.shape[-1]
        print(test_image.shape)
        with torch.no_grad():
            prediction = self.model(test_image.to(device))[0]

        prediction_arr.append(prediction.cpu())
        list_of_hist_images.append((fixed_noise,base_image))

        # Inject mean noise from each class and pass through model
        for i in range(1,len(mean_noise)+1):

            pattern = mean_noise[i-1]
            
            z_new = weights*pattern + (1-weights)*(test_image.cpu())
            list_of_hist_images.append((pattern,z_new.reshape(pixel,pixel)))
            
            with torch.no_grad():
                prediction_prob= self.model(z_new.to(device))[0].to('cpu')
                
            prediction_arr.append(prediction_prob)
            prediction = prediction_prob.max(1)[1]

            misclassification = 1 - (((prediction== i-1).sum()).numpy()/len(prediction))
            print("class: {}, misclassification_rate: {}".format(i-1,np.round(misclassification,3)))
        
        return prediction_arr,list_of_hist_images
        
    def denoising_image_prediction(self,
                                   noise_pattern : torch.Tensor,
                                   test_image : torch.Tensor,
                                   ground_truth : torch.Tensor,
                                   denoiser_lambda : function,
                                   weight_injection : float):
        """Applies a degree of noise into a chosen test image and later applies a denoiser function.
           The function returns a list of clean,noisy and denoised images as well as their predicted class
           The returned arrays are later used with the grid_display function to visually assess the effect of a particular denoiser on an image

        Args:
            noise_pattern (Tensor): mean_noise pattern for a specific class
            test_image (Tensor): chosen test image
            ground_truth (Tensor): classification label for specific image
            denoiser_lambda (lambda(Tensor)): lambda function for a pre-defined denoising method of choice with Tensor image as input
            weight_injection (float): proportion of noise applied to image

        Returns:
            list_of_images (List[Tensor]): The three images (clean, noisy, denoised) are appended in this list
            list_of_names (List[str]) : List of title names to appear as graph header
        """           
    
        list_of_images , list_of_names =[],[]

        # Noise Injection
        injected_noise = weight_injection*noise_pattern + (1-weight_injection)*(test_image.cpu().numpy())
        injected_noise = torch.tensor(injected_noise).cuda()

        # Prediction of injected noise
        with torch.no_grad():
            injected_prediction = self.model(injected_noise[:,None,...])[0].max(1)[1]
        list_of_images.extend((test_image.reshape(28,28).cpu(), injected_noise.reshape(28,28).cpu()))

        # Denoise method is applied, image then passed through model
        denoised_image = denoiser_lambda(injected_noise.squeeze(0).cpu().numpy())
        denoised_image = torch.tensor(denoised_image[None,None,...]).type(torch.FloatTensor).cuda()
        with torch.no_grad():
            denoised_prediction = self.model(denoised_image)[0].max(1)[1]

        list_of_images.extend(denoised_image.squeeze(0).cpu())    
        list_of_names = ["Ground Truth: {}".format(ground_truth.numpy()),
                         "Noisy Prediction :{}".format(injected_prediction[0]),
                         "Denoised Prediction: {}".format(denoised_prediction[0])]
        return list_of_images,list_of_names

    def denoise_mask_transform(self,
                                data : torch.Datasets,
                                pixel : int,
                                noise_pattern : torch.Tensor,
                                noise_weight : float,
                                denoiser_lambda : function,
                                percentile_threshold : float):
        """Applies a denoising method to dataset injected with a mean noise pattern. 
           After denoising, based on a percentile threshold, only some pixel values are kept.
           This creates a mask around the pixels.

        Args:
            data (torch dataset): torch dataset the user desires to denoise.
            pixel (int): size of image
            noise_pattern (Tensor): mean_noise pattern for a specific class
            noise_weight (float): proportion of noise applied to image
            denoiser_lambda (function(Tensor)): lambda function for a pre-defined denoising method of choice with Tensor image as input
            percentile_threshold (float): percentile of pixels the user desires to keep. Values strictly between 0-100.

        Returns:
            markers (ndarray): numpy array of images with mask applied.
        """        
        # Noise Injection 
        injected_noise = noise_weight*noise_pattern + (1 - noise_weight) * (data.cpu().numpy())
        injected_noise = injected_noise.reshape(len(data), pixel, pixel).numpy()
        denoised_img = denoiser_lambda(injected_noise) # Denoise the dataset

        markers = np.zeros((len(data),pixel,pixel))

        # Applies mask based on pixel values
        for image_idx, image in enumerate(denoised_img):
            image = (image - image.min())/(image.max()-image.min())
            equalizer_img = exposure.equalize_adapthist(image) #histogram of pixel values of denoised image
            threshold = np.percentile(equalizer_img, percentile_threshold)
            markers[image_idx][(equalizer_img < threshold)] = 0
            markers[image_idx][(equalizer_img > threshold)] = 1

        return markers