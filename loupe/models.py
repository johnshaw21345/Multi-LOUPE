"""
    LOUPE
    By Cagla Deniz Bahadir, Adrian V. Dalca and Mert R. Sabuncu
    Primary mail: cagladeniz94@gmail.com
    
    Please cite the below paper for the code:
    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Learning-based Optimization of the Under-sampling Pattern in MRI." 
    IPMI 2019
    arXiv preprint arXiv:1901.01960 (2019).
"""

"""
This is not the original code. Modified by Zhikai Yang and Yihan Xiao.
"""

# core python
import sys
import numpy as np
import torch
import torch.nn as nn
# local models
from . import layers
from .unet_parts import UNet
from unet import  get_model

class rec_unet_3ch(torch.nn.Module):
    def __init__(self, n_channels, n_classes=2, bilinear=True):
        super(rec_unet_3ch, self).__init__()
        self.n_channels = n_channels
        self.unet = UNet(self.n_channels, n_classes, bilinear) 
        
    def forward(self, inputs, masks):
        last_tensor = inputs
        last_tensor_mask = masks

        last_tensor_mask_1= last_tensor_mask[0]
        last_tensor_mask_2= last_tensor_mask[1]
        last_tensor_mask_3= last_tensor_mask[2]

        # Assertion error when directly calling layers.FFT()?

        last_tensor_1 = torch.stack((last_tensor[:,0,:,:][:,np.newaxis],last_tensor[:,3,:,:][:,np.newaxis]),dim=1)
        last_tensor_1 = torch.squeeze(last_tensor_1)
        if(last_tensor_1.dim() < 4):
            last_tensor_1 = last_tensor_1[np.newaxis,:] # [32, 2, 200, 200]

        #print(last_tensor_1.shape)

        last_tensor_2 = torch.stack((last_tensor[:,1,:,:][:,np.newaxis],last_tensor[:,4,:,:][:,np.newaxis]),dim=1)
        last_tensor_2 = torch.squeeze(last_tensor_2)
        if(last_tensor_2.dim() < 4):
            last_tensor_2 = last_tensor_2[np.newaxis,:] # [32, 2, 200, 200]

        last_tensor_3 = torch.stack((last_tensor[:,2,:,:][:,np.newaxis],last_tensor[:,5,:,:][:,np.newaxis]),dim=1)
        last_tensor_3 = torch.squeeze(last_tensor_3)
        if(last_tensor_3.dim() < 4):
            last_tensor_3 = last_tensor_3[np.newaxis,:] # [32, 2, 200, 200]

        #last_tensor_1 = torch.squeeze(torch.stack([last_tensor[:,0,:,:][:,np.newaxis],last_tensor[:,3,:,:][:,np.newaxis]],dim=1))
        #last_tensor_2 = torch.squeeze(torch.stack([last_tensor[:,1,:,:][:,np.newaxis],last_tensor[:,4,:,:][:,np.newaxis]],dim=1))
        #last_tensor_3 = torch.squeeze(torch.stack([last_tensor[:,2,:,:][:,np.newaxis],last_tensor[:,5,:,:][:,np.newaxis]],dim=1))   

        # Assertion error when directly calling layers.FFT()?

        last_tensor_1 = layers.FFT()(last_tensor_1)
        last_tensor_2 = layers.FFT()(last_tensor_2)
        last_tensor_3 = layers.FFT()(last_tensor_3)



        last_tensor_1_k = layers.UnderSample()(last_tensor_1, last_tensor_mask_1)
        last_tensor_1 = layers.IFFT()(last_tensor_1_k)
        last_tensor_2_k = layers.UnderSample()(last_tensor_2, last_tensor_mask_2)
        last_tensor_2 = layers.IFFT()(last_tensor_2_k)
        last_tensor_3_k = layers.UnderSample()(last_tensor_3, last_tensor_mask_3)
        last_tensor_3 = layers.IFFT()(last_tensor_3_k)

        last_tensor_cat = torch.stack((last_tensor_1[:,0,:,:],last_tensor_2[:,0,:,:],last_tensor_3[:,0,:,:],\
            last_tensor_1[:,1,:,:],last_tensor_2[:,1,:,:],last_tensor_3[:,1,:,:]),dim=1)
        
        unet_tensor = self.unet(last_tensor_cat)       

        #print(unet_tensor.shape)

        abs_tensor1 = last_tensor_1
        abs_tensor2 = last_tensor_2
        abs_tensor3 = last_tensor_3
        abs_tensor_cat = torch.stack((abs_tensor1[:,0,:,:],abs_tensor2[:,0,:,:],abs_tensor3[:,0,:,:],\
                                    abs_tensor1[:,1,:,:],abs_tensor2[:,1,:,:],abs_tensor3[:,1,:,:]),dim=1)
        
        unet_tensor = abs_tensor_cat + unet_tensor

        #print(unet_tensor.shape)

        unet_tensor1 = torch.stack([unet_tensor[:,0,:,:],unet_tensor[:,3,:,:]],dim=1)
        unet_tensor2 = torch.stack([unet_tensor[:,1,:,:],unet_tensor[:,4,:,:]],dim=1)
        unet_tensor3 = torch.stack([unet_tensor[:,2,:,:],unet_tensor[:,5,:,:]],dim=1)

        unet_tensor = torch.stack([layers.ComplexAbs()(unet_tensor1),layers.ComplexAbs()(unet_tensor2),\
                        layers.ComplexAbs()(unet_tensor3)],dim=1)
        
        #print(unet_tensor.shape)
        
        unet_tensor = torch.squeeze(unet_tensor)    

        outputs = unet_tensor
        return outputs
    
class rec_unet_1ch(torch.nn.Module):
    def __init__(self, n_channels, n_classes=2, bilinear=True):
        super(rec_unet_1ch, self).__init__()
        self.n_channels = n_channels
        self.unet = UNet(2, 2, bilinear) 
        
    def forward(self, inputs, masks):
        last_tensor = inputs
        last_tensor_mask = masks

        last_tensor_mask_1= last_tensor_mask[0]

        # Assertion error when directly calling layers.FFT()?

        last_tensor_1 = torch.stack((last_tensor[:,0,:,:][:,np.newaxis],last_tensor[:,1,:,:][:,np.newaxis]),dim=1)
        last_tensor_1 = torch.squeeze(last_tensor_1)
        if(last_tensor_1.dim() < 4):
            last_tensor_1 = last_tensor_1[np.newaxis,:] # [32, 2, 200, 200]

        #print(last_tensor_1.shape)

        last_tensor_1 = layers.FFT()(last_tensor_1)

        last_tensor_1_k = layers.UnderSample()(last_tensor_1, last_tensor_mask_1)
        last_tensor_1 = layers.IFFT()(last_tensor_1_k)

        #print(last_tensor_1.shape)

        last_tensor_cat = torch.stack((last_tensor_1[:,0,:,:], last_tensor_1[:,1,:,:]),dim=1)  # [32,2,200,200]
        
        unet_tensor = self.unet(last_tensor_cat)     
        #print(unet_tensor.shape)  

        abs_tensor1 = last_tensor_1

        abs_tensor_cat = torch.stack((abs_tensor1[:,0,:,:], abs_tensor1[:,1,:,:]),dim=1)
        
        unet_tensor = abs_tensor_cat + unet_tensor

        #print(unet_tensor.shape)

        unet_tensor1 = torch.stack([unet_tensor[:,0,:,:],unet_tensor[:,1,:,:]],dim=1) # [32,2,200,200]

        unet_tensor = layers.ComplexAbs()(unet_tensor1)
        
        #print(unet_tensor.shape)  

        outputs = unet_tensor
        return outputs
    
class loupe_3ch(torch.nn.Module):
    def __init__(self, n_channels, n_classes=2, sparsity=0.25, bilinear=True):
        super(loupe_3ch, self).__init__()
        self.n_channels = n_channels
        self.pmask_slope=200
        self.pmask_init=None
        self.sample_slope=200
        self.sparsity = sparsity
        self.ProbMask1 = layers.ProbMask(self.pmask_slope, initializer=self.pmask_init,shape = (1,200,200))
        self.ProbMask2 = layers.ProbMask(self.pmask_slope, initializer=self.pmask_init,shape = (1,200,200))
        self.ProbMask3 = layers.ProbMask(self.pmask_slope, initializer=self.pmask_init,shape = (1,200,200))

        self.last_tensor_mask_1 = layers.ThresholdRandomMask(self.sample_slope)
        self.last_tensor_mask_2 = layers.ThresholdRandomMask(self.sample_slope)
        self.last_tensor_mask_3 = layers.ThresholdRandomMask(self.sample_slope)
        self.unet = UNet(6, 6, bilinear) 
        
    def forward(self, inputs):
        last_tensor = inputs

        # Assertion error when directly calling layers.FFT()?

        last_tensor_1 = torch.stack((last_tensor[:,0,:,:][:,np.newaxis],last_tensor[:,3,:,:][:,np.newaxis]),dim=1)
        last_tensor_1 = torch.squeeze(last_tensor_1)
        if(last_tensor_1.dim() < 4):
            last_tensor_1 = last_tensor_1[np.newaxis,:] # [1,200,200,3]

        #print(last_tensor_1.shape)

        last_tensor_2 = torch.stack((last_tensor[:,1,:,:][:,np.newaxis],last_tensor[:,4,:,:][:,np.newaxis]),dim=1)
        last_tensor_2 = torch.squeeze(last_tensor_2)
        if(last_tensor_2.dim() < 4):
            last_tensor_2 = last_tensor_2[np.newaxis,:] # [1,200,200,3]

        last_tensor_3 = torch.stack((last_tensor[:,2,:,:][:,np.newaxis],last_tensor[:,5,:,:][:,np.newaxis]),dim=1)
        last_tensor_3 = torch.squeeze(last_tensor_3)
        if(last_tensor_3.dim() < 4):
            last_tensor_3 = last_tensor_3[np.newaxis,:] # [1,200,200,3]

        #last_tensor_1 = torch.squeeze(torch.stack([last_tensor[:,0,:,:][:,np.newaxis],last_tensor[:,3,:,:][:,np.newaxis]],dim=1))
        #last_tensor_2 = torch.squeeze(torch.stack([last_tensor[:,1,:,:][:,np.newaxis],last_tensor[:,4,:,:][:,np.newaxis]],dim=1))
        #last_tensor_3 = torch.squeeze(torch.stack([last_tensor[:,2,:,:][:,np.newaxis],last_tensor[:,5,:,:][:,np.newaxis]],dim=1))   

        # Assertion error when directly calling layers.FFT()?

        last_tensor_1 = layers.FFT()(last_tensor_1)
        last_tensor_2 = layers.FFT()(last_tensor_2)
        last_tensor_3 = layers.FFT()(last_tensor_3)

        prob_mask_tensor_1 = self.ProbMask1(last_tensor_1)
        prob_mask_tensor_2 = self.ProbMask2(last_tensor_2)
        prob_mask_tensor_3 = self.ProbMask3(last_tensor_3)
        
        prob_mask_tensor_1 = layers.RescaleProbMap(self.sparsity)(prob_mask_tensor_1)
        prob_mask_tensor_2 = layers.RescaleProbMap(self.sparsity)(prob_mask_tensor_2)
        prob_mask_tensor_3 = layers.RescaleProbMap(self.sparsity)(prob_mask_tensor_3)
                        
        thresh_tensor_1 = layers.RandomMask()(prob_mask_tensor_1)
        thresh_tensor_2 = layers.RandomMask()(prob_mask_tensor_2)
        thresh_tensor_3 = layers.RandomMask()(prob_mask_tensor_3)
        
        last_tensor_mask_1 = self.last_tensor_mask_1([prob_mask_tensor_1, thresh_tensor_1])
        last_tensor_mask_2 = self.last_tensor_mask_2([prob_mask_tensor_2, thresh_tensor_2])
        last_tensor_mask_3 = self.last_tensor_mask_3([prob_mask_tensor_3, thresh_tensor_3])

        last_tensor_1_k = layers.UnderSample()(last_tensor_1, last_tensor_mask_1)
        last_tensor_1 = layers.IFFT()(last_tensor_1_k)
        last_tensor_2_k = layers.UnderSample()(last_tensor_2, last_tensor_mask_2)
        last_tensor_2 = layers.IFFT()(last_tensor_2_k)
        last_tensor_3_k = layers.UnderSample()(last_tensor_3, last_tensor_mask_3)
        last_tensor_3 = layers.IFFT()(last_tensor_3_k)

        last_tensor_mask_cat = torch.stack((last_tensor_mask_1,last_tensor_mask_2,last_tensor_mask_3),dim=-1)
        
        last_tensor_mask_cat = torch.squeeze(last_tensor_mask_cat)

        if(last_tensor_mask_cat.dim() < 4):
            last_tensor_mask_cat = last_tensor_mask_cat[np.newaxis,:] # [1,200,200,3]

        last_tensor_mask_cat = last_tensor_mask_cat.permute(0,3,1,2) # [1,200,200,3] -> [1,3,200,200]

        #print(unet_tensor.shape)

        last_tensor_cat = torch.stack((last_tensor_1[:,0,:,:],last_tensor_2[:,0,:,:],last_tensor_3[:,0,:,:],\
                                       last_tensor_1[:,1,:,:],last_tensor_2[:,1,:,:],last_tensor_3[:,1,:,:]),dim=1) 
        
        unet_tensor = self.unet(last_tensor_cat)

        abs_tensor1 = last_tensor_1
        abs_tensor2 = last_tensor_2
        abs_tensor3 = last_tensor_3
        abs_tensor_cat = torch.stack((abs_tensor1[:,0,:,:],abs_tensor2[:,0,:,:],abs_tensor3[:,0,:,:],\
                                    abs_tensor1[:,1,:,:],abs_tensor2[:,1,:,:],abs_tensor3[:,1,:,:]),dim=1)
        
        unet_tensor = abs_tensor_cat + unet_tensor

        #print(unet_tensor.shape)

        unet_tensor1 = torch.stack([unet_tensor[:,0,:,:],unet_tensor[:,3,:,:]],dim=1)
        unet_tensor2 = torch.stack([unet_tensor[:,1,:,:],unet_tensor[:,4,:,:]],dim=1)
        unet_tensor3 = torch.stack([unet_tensor[:,2,:,:],unet_tensor[:,5,:,:]],dim=1)

        unet_tensor = torch.stack([layers.ComplexAbs()(unet_tensor1),layers.ComplexAbs()(unet_tensor2),\
                        layers.ComplexAbs()(unet_tensor3)],dim=1)
        
        #print(unet_tensor.shape)

        prob_mask_tensor_cat = torch.stack((prob_mask_tensor_1,prob_mask_tensor_2,prob_mask_tensor_3),dim=-1)
        
        prob_mask_tensor_cat = torch.squeeze(prob_mask_tensor_cat)

        if(prob_mask_tensor_cat.dim() < 4):
            prob_mask_tensor_cat = prob_mask_tensor_cat[np.newaxis,:] # [32,200,200,8]

        prob_mask_tensor_cat = prob_mask_tensor_cat.permute(0,3,1,2) # [32,200,200,8] -> [32,8,200,200]
        
        unet_tensor = torch.squeeze(unet_tensor)

        #print(unet_tensor.shape)

        outputs = [unet_tensor,last_tensor_mask_cat,prob_mask_tensor_cat]
        return outputs

class loupe_1ch(torch.nn.Module):
    def __init__(self, n_channels, n_classes=2, sparsity=0.25, bilinear=True):
        super(loupe_1ch, self).__init__()
        self.n_channels = n_channels
        self.pmask_slope=200
        self.pmask_init=None
        self.sample_slope=200
        self.sparsity = sparsity # 0.15
        self.ProbMask1 = layers.ProbMask(self.pmask_slope, initializer=self.pmask_init,shape = (1,200,200))

        self.last_tensor_mask_1 = layers.ThresholdRandomMask(self.sample_slope)

        self.unet = UNet(2, 2, bilinear) 
        
    def forward(self, inputs):
        last_tensor = inputs # [32,1,200,200]

        # Assertion error when directly calling layers.FFT()?

        last_tensor_1 = torch.stack((last_tensor[:,0,:,:][:,np.newaxis],last_tensor[:,1,:,:][:,np.newaxis]),dim=1) # [32,2,200,200]
        last_tensor_1 = torch.squeeze(last_tensor_1)
        if(last_tensor_1.dim() < 4):
            last_tensor_1 = last_tensor_1[np.newaxis,:]


        last_tensor_1 = layers.FFT()(last_tensor_1) # [32,2,200,200]

        prob_mask_tensor_1 = self.ProbMask1(last_tensor_1) # [32,2,200,200]
        
        prob_mask_tensor_1 = layers.RescaleProbMap(self.sparsity)(prob_mask_tensor_1) # [32,2,200,200]
                        
        thresh_tensor_1 = layers.RandomMask()(prob_mask_tensor_1) # [32,2,200,200]
        
        last_tensor_mask_1 = self.last_tensor_mask_1([prob_mask_tensor_1, thresh_tensor_1]) # [32,1,200,200]

        last_tensor_1_k = layers.UnderSample()(last_tensor_1, last_tensor_mask_1) # [32,2,200,200]
        last_tensor_1 = layers.IFFT()(last_tensor_1_k) # [32,2,200,200]

        last_tensor_mask_cat = last_tensor_mask_1 # [32,1,200,200]

        #print(unet_tensor.shape)

        last_tensor_cat = torch.stack((last_tensor_1[:,0,:,:], last_tensor_1[:,1,:,:]),dim=1)  # [32,2,200,200]
        
        unet_tensor = self.unet(last_tensor_cat)

        abs_tensor1 = last_tensor_1

        abs_tensor_cat = torch.stack((abs_tensor1[:,0,:,:], abs_tensor1[:,1,:,:]),dim=1)
        
        unet_tensor = abs_tensor_cat + unet_tensor

        #print(unet_tensor.shape)

        unet_tensor1 = torch.stack([unet_tensor[:,0,:,:],unet_tensor[:,1,:,:]],dim=1) # [32,2,200,200]

        unet_tensor = layers.ComplexAbs()(unet_tensor1)
        
        #print(unet_tensor.shape)

        prob_mask_tensor_cat = prob_mask_tensor_1

        #print(prob_mask_tensor_cat.shape)

        #print(unet_tensor.shape)

        outputs = [unet_tensor,last_tensor_mask_cat,prob_mask_tensor_cat]
        return outputs