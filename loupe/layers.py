"""
    Layers for LOUPE
    
    For more details, please read:
    
    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Learning-based Optimization of the Under-sampling Pattern in MRI." 
    IPMI 2019. https://arxiv.org/abs/1901.01960.
"""

"""
    class FFT is modified.
    The redundant fftshift is removed to fit the new dataset.
"""

# third party
import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

def binary_top_n(matrix,n):
    flattened_matrix = matrix.flatten()  # Flatten the matrix into a 1D array
    num_elements = len(flattened_matrix)
    num_top_values = int(num_elements * n)  # Calculate the number of top values (10% of total elements)
    top_indices = np.argsort(flattened_matrix)[-num_top_values:]  # Get the indices of the top values
    
    binary_matrix = np.zeros_like(matrix)  # Create a binary matrix of the same shape as the input matrix
    binary_matrix.ravel()[top_indices] = 1  # Set the values at top indices to 1
    
    return binary_matrix


class RescaleProbMap(nn.Module):
    def __init__(self, sparsity):
        super(RescaleProbMap, self).__init__()
        self.sparsity = sparsity

    def forward(self, x):
        xbar = torch.mean(x)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)
        le = torch.le(r, 1).float()
        return le * x * r + (1 - le) * (1 - (1 - x) * beta)

class ProbMask(nn.Module):
    def __init__(self, slope=10, initializer=None,shape=(1,32,32)):
        super(ProbMask, self).__init__()
        if initializer is None:
            self.initializer = self._logit_slope_random_uniform
        else:
            self.initializer = initializer
        self.slope = slope
        self.mult = nn.Parameter(self._logit_slope_random_uniform(shape), requires_grad=True)

    def forward(self, x):
        # logit_weights = 0 * x[..., 0:1] + self.mult
        logit_weights = 0 * x[:, 0:1,:,:] + self.mult
        return torch.sigmoid(self.slope * logit_weights)

    def _logit_slope_random_uniform(self, shape, dtype=None, eps=0.01):
        x = torch.rand(shape, dtype=dtype) * (1.0 - 2 * eps) + eps
        return -torch.log(1.0 / x - 1.0) / self.slope
    

class ThresholdRandomMask(nn.Module):
    """ 
    Local thresholding layer
    Takes as input the input to be thresholded, and the threshold
    Modified from Local Linear Layer code in https://github.com/adalca/neuron
    """

    def __init__(self, slope=12):
        super(ThresholdRandomMask, self).__init__()
        self.slope = None
        if slope is not None:
            self.slope = nn.Parameter(torch.tensor(slope, dtype=torch.float32))

    def forward(self, x):
        inputs, thresh = x[0], x[1]
        if self.slope is not None:
            return torch.sigmoid(self.slope * (inputs - thresh))
        else:
            return inputs > thresh


class RandomMask(nn.Module):
    """ 
    Create a random binary mask of the same size as the input shape
    """

    def __init__(self):
        super(RandomMask, self).__init__()
        
    def forward(self, x):
        input_shape = x.shape
        threshs = torch.rand(input_shape, dtype=torch.float32).cuda()
        return torch.zeros_like(x) + threshs
    
    

class FFT(torch.nn.Module):
    """
    fft layer, assuming the real/imag are input/output via two features

    Input: torch.float32 of size [batch_size, ..., 2]
    Output: torch.float32 of size [batch_size, ..., 2]
    """

    def __init__(self):
        super(FFT, self).__init__()

    def forward(self, inputx):
        #assert inputx.shape[1] == 2, 'input has to have two features'

        # get the right fft
        if len(inputx.shape) - 2 == 1:
            fft_fn = fft.fft
        elif len(inputx.shape) - 2 == 2:
            fft_fn = fft.fft2
        else:
            fft_fn = fft.fftn
        complex_img = torch.complex(inputx[:, 0,:,:], inputx[:, 1,:,:])
        tmp = fft.fftshift(complex_img,(-2,-1))
        fft_im = fft_fn(tmp,norm='backward')
        #fft_im = fft.fftshift(fft_im,(-2,-1)) # This step affects fftshift, disable to put low frequncy at center.

        # go back to two-feature representation
        fft_im = torch.stack([torch.real(fft_im), torch.imag(fft_im)], dim=1)
        return fft_im.float()

    def compute_output_shape(self, input_shape):
        return input_shape

class IFFT(torch.nn.Module):
    """
    ifft layer, assuming the real/imag are input/output via two features
    Input: torch.float32 of size [batch_size, ..., 2]
    Output: torch.float32 of size [batch_size, ..., 2]
    """

    def __init__(self):
        super(IFFT, self).__init__()

    def forward(self, inputx):
        #assert inputx.shape[1] == 2, 'input has to have two features'
        # get the right fft
        if len(inputx.shape) - 2 == 1:
            ifft_fn = fft.ifft
        elif len(inputx.shape) - 2 == 2: #This is for 3 channels
            ifft_fn = fft.ifft2
        else:
            ifft_fn = fft.ifftn

        # get ifft complex image
        complex_img = torch.complex(inputx[:, 0,:,:], inputx[:, 1,:,:])
        tmp = fft.ifftshift(complex_img,(-2,-1))
        ifft_im = ifft_fn(tmp,norm='backward')
        ifft_im = fft.ifftshift(ifft_im,(-2,-1))

        # go back to two-feature representation
        ifft_im = torch.stack([torch.real(ifft_im), torch.imag(ifft_im)], dim=1)
        return ifft_im.float()

    def compute_output_shape(self, input_shape):
        return input_shape
    
class ConcatenateZero(nn.Module):
    def __init__(self):
        super(ConcatenateZero, self).__init__()

    def forward(self, x):
        zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3],  dtype=x.dtype, device=x.device)
        return torch.cat([x, zeros], dim=1)
    
class UnderSample(nn.Module):
    def __init__(self):
        super(UnderSample, self).__init__()

    def forward(self, x, mask):
        mask = mask.expand_as(x)
        return x * mask


class ComplexAbs(nn.Module):
    def __init__(self):
        super(ComplexAbs, self).__init__()

    def forward(self, x):
        out = torch.abs(torch.complex(x[:, 0,:,:], x[:, 1,:,:]))
        out = torch.unsqueeze(out,1)
        return out