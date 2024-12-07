from cv2 import norm
import numpy as np
import torch
from parameters import Parameters
from torch.fft import fftshift, ifftshift, fft2,ifft2
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import copy
from sklearn.model_selection import train_test_split
params = Parameters()

def net_init(net):
    def weight_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.kaiming_uniform_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.kaiming_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
    for m in net.modules():
        weight_init(m)

def normalize_batch_torch(p, return_mv=False,mean=False, std=False):
    ''' normalize each slice alone'''
    if torch.std(p) == 0:
        raise ZeroDivisionError
    shape = p.shape
    if p.ndimension() == 4:
        pv = p.reshape([shape[0], shape[1], shape[2] * shape[3]])
    else:
        raise NotImplementedError

    if shape[1]==2:
        p_new = torch.zeros(p.shape)
        if mean:
            p_new[:,0,:,:] = (p[:,0,:,:] - mean.real) / std.real
            p_new[:,1,:,:] = (p[:,1,:,:] - mean.imag) / std.imag            
        else:
            mean = pv.mean(dim=2, keepdim=True).unsqueeze(p.ndimension() - 1)
            std = pv.std(dim=2, keepdim=True).unsqueeze(p.ndimension() - 1)
            p_new = (p - mean) / std
        p_new = p_new.reshape([shape[0], shape[1], shape[2] ,shape[3]])
        if return_mv:
            return p_new, mean, std
        return p_new
    else:
        if mean:
            p_new = (p - mean) / std
        else:
            mean = pv.mean(dim=2, keepdim=True).unsqueeze(p.ndimension() - 1)
            std = pv.std(dim=2, keepdim=True).unsqueeze(p.ndimension() - 1)
            p_new = (p - mean) / std
        p_new = p_new.reshape([shape[0], shape[1], shape[2] ,shape[3]])
        if return_mv:
            return p_new, mean, std
        return p_new


def unnormalize_batch_torch(p, mean, std):
    ''' normalize each slice alone'''
    shape = p.shape
    if p.ndimension() == 4:
        pv = p.reshape([shape[0], shape[1], shape[2] * shape[3]])
    else:
        raise NotImplementedError
    p_new = torch.zeros(p.shape)
    if shape[1]==2:
        p_new[:,0,:,:] = (p[:,0,:,:] + mean.real) * std.real
        p_new[:,1,:,:] = (p[:,1,:,:] + mean.imag) * std.imag
    else:
        p_new = (p + mean) * std
    p_new = p_new.reshape([shape[0], shape[1], shape[2] ,shape[3]])
    return p_new

def get_magnitude(input):
    return (input[:, :, :, :, 0] ** 2 + input[:, :, :, :, 1] ** 2) ** 0.5
    # return np.abs(input)

def ifft_my(img,axes):
    '''
    input: k complex form [n c w h]
    return: [w h] complex form
    '''
    # print("len(img.shape)",len(img.shape))
    if len(img.shape)==4 and img.shape[1]==2:
        tmp = torch.complex(img[:,0,:,:], img[:,1 ,:,:])
        tmp = ifftshift(tmp,axes)
        tmp = ifft2(tmp, norm='backward')
        tmp = ifftshift(tmp,axes)
        # print("tmp",tmp.shape)
    else:
        tmp = ifftshift(img,axes)
        tmp = ifft2(tmp, norm='backward')
        tmp = ifftshift(tmp,axes)
    return tmp

def fft_my(img,axes):
    '''
    img complex form [w h] or [n w h]
    return: [w h] complex form or [n w h]
    '''
    if len(img.shape)==4 and img.shape[1]==2:
        tmp = torch.complex(img[:,0,:,:], img[:,1,:,:])
        tmp = fftshift(tmp,axes)
        tmp = fft2(tmp, norm='backward')
        tmp = fftshift(tmp,axes)
    else:
        tmp = fftshift(img,axes)
        tmp = fft2(tmp,norm='backward')
        tmp = fftshift(tmp,axes)
    return tmp


def ifft_batch(input, save=False):
    '''
    input shape [n 1 w h 1] or [n 1 w h 2]  dim=5 or dim=4 [N 1 w h]
    return shape same 
    '''
    batch_size = input.shape[0]
    input  = torch.squeeze(input)
    if batch_size==1:
        input = torch.unsqueeze(input,dim=0)
    input = input.detach().cpu()
    # print("input.shape",input.shape)
    # print("len(input.shape)",len(input.shape))
    if len(input.shape) == 4 :
        if input.shape[-1]==2:
            img = np.zeros(input.shape[:-1])
            for i in range(input.shape[0]):
                tmp = torch.zeros(input[i].shape[:-1], dtype=torch.complex64)
                tmp.real = input[i][:, :, 0]
                tmp.imag = input[i][:, :, 1]
                ori = ifft_my(tmp,(-2,-1))
                img[i] = np.abs(ori)
    if len(input.shape) == 3 :
            img = np.zeros(input.shape)
            for i in range(input.shape[0]):
                if save:
                    plt.matshow(np.abs(input[i]))
                    plt.savefig("before_out.png")
                ori = ifft_my(tmp,(-2,-1))
                if save:
                    plt.matshow(np.abs(ori))
                    plt.savefig("out.png")
                img[i] = np.abs(ori)
    re = torch.unsqueeze(torch.from_numpy(img).cuda(),1)
    return re

def ifft_cpx_batch(input, save=False):
    '''
    input shape  [n 1 w h 2] 
    return shape [n 1 w h 2] the last one is real and imagary part 
    '''
    batch_size = input.shape[0]
    input  = torch.squeeze(input)
    if batch_size==1:
        input = torch.unsqueeze(input,dim=0)
    # input = input.detach().cpu().numpy()
    # print("input.shape",input.shape)
    # print("len(input.shape)",len(input.shape))
    if len(input.shape) == 4 :
        if input.shape[-1]==2:
            img = np.zeros(input.shape[:-1],dtype=np.complex64)
            for i in range(input.shape[0]):
                tmp = np.zeros(input[i].shape[:-1], dtype=np.complex64)
                # print("input[i][:, :, 0]",input[i][:, :, 0].shape)
                tmp.real = input[i][:, :, 0]
                tmp.imag = input[i][:, :, 1]
                ori = ifft_my(tmp,(-2,-1))
                img[i] = ori
    if len(input.shape) == 3 :
            img = np.zeros(input.shape)
            for i in range(input.shape[0]):
                if save:
                    plt.matshow(np.abs(input[i]))
                    plt.savefig("before_out.png")                
                ori = ifft_my(input[i],(-2,-1))
                if save:
                    plt.matshow(np.abs(ori))
                    plt.savefig("out.png")
                img[i] = ori
    re = torch.unsqueeze(torch.from_numpy(img).cuda(),1)
    zz = torch.stack([re.real, re.imag],dim=4)
    return zz


def ifft_cpx_batch_realtwo(input, save=False):
    '''
    input shape  [n 1 w h 2] 
    return shape [n 2 w h] the 2nd is real and imagary part 
    '''
    batch_size = input.shape[0]
    input  = torch.squeeze(input)
    if batch_size==1:
        input = torch.unsqueeze(input,dim=0)
    # input = input.detach().cpu().numpy()
    # print("input.shape",input.shape)
    # print("len(input.shape)",len(input.shape))
    if len(input.shape) == 4 :
        if input.shape[-1]==2:
            img = np.zeros(input.shape[:-1],dtype=np.complex64)
            for i in range(input.shape[0]):
                tmp = np.zeros(input[i].shape[:-1], dtype=np.complex64)
                # print("input[i][:, :, 0]",input[i][:, :, 0].shape)
                tmp.real = input[i][:, :, 0]
                tmp.imag = input[i][:, :, 1]
                ori = ifft_my(tmp,(-2,-1))
                img[i] = ori
    if len(input.shape) == 3 :
            img = np.zeros(input.shape)
            for i in range(input.shape[0]):
                # print("input[i].shape",input[i].shape,"input[i].dtype",input[i].dtype)
                if save:
                    plt.matshow(np.abs(input[i]))
                    plt.savefig("before_out.png")                
                ori = ifft_my(input[i],(-2,-1))
                # print("ori[i].shape",ori.shape,"ori[i].dtype",ori.dtype)
                if save:
                    plt.matshow(np.abs(ori))
                    plt.savefig("out.png")
                img[i] = ori
    re = torch.from_numpy(img).cuda()
    zz = torch.stack([re.real, re.imag],dim=1)
    return zz


def ifft_single(input):
    '''
    input shape [n 1 w h 1] or [n 1 w h 2]  dim=5 or dim=4 [1 w h 1] or [1 w h 2]
    return shape same   
    '''
    input  = torch.squeeze(input).detach().cpu().numpy()
    if len(input.shape)==2:
        ori = ifft_my(tmp,(-2,-1))
        return torch.from_numpy(ori.real)      
    if input.shape[-1]==2:
        img = np.zeros(input.shape[:-1])
        tmp = np.zeros(input.shape[:-1], dtype=np.complex64)
        tmp.real = input[:, :, 0]
        tmp.imag = input[:, :, 1]
        ori = ifft_my(tmp,(-2,-1))
        return torch.from_numpy(ori.real)
    # re = torch.unsqueeze(torch.from_numpy(img).cuda(),1)
    # return re


def create_my_data(params, view):
    """Create data basedset for training and validation.
    Note that in practice, at test time the method will need to be applied to
    the whole volume. In addition, one would need more data to prevent
    overfitting.
    """
    path = os.path.join(params.train_path, view, "train_data")
    files = os.listdir(path)
    inputs = []
    outs = []    
    if view == "human":
        for file in files:
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = input
            inputs.append(input)
            outs.append(out)
    else:
        for file in files:
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = loadmat(os.path.join(path,file))['kSpc']
            inputs.append(input)
            outs.append(out)        

    data_in = np.array(inputs)
    data_out = np.array(outs)
    # transpose shape from [n,96,96,73] -> [n,73,96,96] then reshape to [n*73,96,96]
    data_in = np.transpose(data_in,(0,3,1,2))
    data_out = np.transpose(data_out,(0,3,1,2))
    data_in = data_in.reshape([-1,96,96])
    data_out = data_out.reshape([-1,96,96])

    print("data_out.shape",data_out.shape)
    if params.mask_type=="rad":
        # x4 is radx4 x8 x16 are d 
        mask = loadmat(params.rad_path)["d"]
        masks = np.repeat(mask[np.newaxis],data_in.shape[0],axis=0)
        data_in = masks * data_out
        print("rad sampling")
    if params.mask_type=="rect":
        mask = np.zeros([96,96]) # [n w w]
        mask[36:60] = 1
        masks = np.repeat(mask[np.newaxis],data_in.shape[0],axis=0)
        data_in = masks * data_in      
    trainx, valx, trainy, valy = train_test_split( data_in, data_out, train_size=params.training_percent, random_state=42)
    train_mean = np.mean(trainx)
    train_std = np.std(trainx)
    print("trainx.shape",trainx.shape)
    print("valx.shape",valx.shape)
    return trainx, trainy, valx, valy, train_mean, train_std

    
def create_my_datathreechannle(params, view):
    """Create data basedset for training and validation.
    Note that in practice, at test time the method will need to be applied to
    the whole volume. In addition, one would need more data to prevent
    overfitting.
    """
    path = os.path.join(params.train_path, view, "train_data")
    files = os.listdir(path)
    inputs = []
    outs = []
    if view=='human':
        for file in files: 
            # print(os.path.join(path,file))
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = input
            inputs.append(input)
            outs.append(out)        
    else:
        for file in files: 
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = loadmat(os.path.join(path,file))['kSpc']
            inputs.append(input)
            outs.append(out)

    data_in = np.array(inputs)
    data_out = np.array(outs)
    # print(view, "data_in.shape",data_in.shape)
    # transpose shape from [n,96,96,73] -> [n,73,96,96] then reshape to [n*73,96,96]
    data_in    = np.transpose(data_out,(0,3,1,2))
    data_out   = np.transpose(data_out,(0,3,1,2))
    data_c_in  = np.zeros((data_in.shape[0],data_in.shape[1],3,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)
    data_c_out = np.zeros((data_in.shape[0],data_in.shape[1],1,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)

    for i in range(data_in.shape[0]):
        # print("[data_in[i][-1][np.newaxis]", data_in[i][-1][np.newaxis].shape, data_in[i][ :2].shape)
        data_c_in[i][0]  = np.stack([data_in[i][-1], data_in[i][0], data_in[i][1] ] )
        for j in range(1,data_in.shape[1]-1):
            data_c_in[i][j] = data_in[i][j-1:j+2]
        data_c_in[i][-1] = np.stack([data_in[i][-2], data_in[i][-1], data_in[i][0]])

        data_c_out[i][0] = data_out[i][0]
        for j in range(1,data_in.shape[1]-1):
            data_c_out[i][j] = data_out[i][j]
        data_c_out[i][-1] = data_out[i][-1]

    data_c_in = data_c_in.reshape([-1,3,96,96])
    data_out_final = data_c_out.reshape([-1,1,96,96])
    # print("data_in.shape",data_in.shape)
    # print("data_out.shape",data_out.shape)
    if params.mask_type=="rad":
        # mask = loadmat(params.threerad_path)["d"]
        mask = loadmat(params.fiverad_path)["d"]
        mask = mask[:,:,1:4]
        # print("mask.shape",mask.shape)        
        # plt.imshow(mask[:,:,0],cmap="gray")
        # plt.savefig("mask_test1.png")
        # plt.imshow(mask[:,:,1],cmap="gray")
        # plt.savefig("mask_test2.png")
        # plt.imshow(mask[:,:,2],cmap="gray")
        # plt.savefig("mask_test3.png")
        mask = np.transpose(mask,(2,0,1))
        print("mask.shape",mask.shape)
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in
        print("rad sampling")
    if params.mask_type=="rect":
        mask = np.zeros([3,96,96]) # [n w w]
        mask[:,36:60] = 1
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in        
    if params.mask_type=="loupe":
        mask = np.load(params.loupe_pattern_path)
        print("mask.shape",mask.shape)
        # plt.imshow(mask[0])
        # plt.savefig("mask_test1.png")
        # plt.imshow(mask[1])
        # plt.savefig("mask_test2.png")
        # plt.imshow(mask[2])
        # plt.savefig("mask_test3.png")
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in
    if params.mask_type=="full":
        data_in = data_c_in
        mask = np.ones((3,96,96))
    # adjancet_mask = (mask[0] + mask[2] + mask[1]) >=1
    # plt.imshow(adjancet_mask)
    # plt.savefig("three_adjacent.png")        
    # masks_sum = (mask[0] + mask[1] + mask[2]) >=1 
    # print("sum_ratio",np.sum(masks_sum)/(masks_sum.shape[1]**2))
    print("data_in",data_in.shape)
    trainx, valx, trainy, valy = train_test_split(data_in, data_out_final, train_size=params.training_percent, random_state=42)
    train_mean = np.mean(trainx)
    train_std = np.std(trainx)
    print("trainx.shape",trainx.shape)
    print("valx.shape",valx.shape)
    return trainx, trainy, valx, valy, train_mean, train_std


def create_my_datafivechannle(params, view):
    """Create data basedset for training and validation.
    Note that in practice, at test time the method will need to be applied to
    the whole volume. In addition, one would need more data to prevent
    overfitting.
    """
    path = os.path.join(params.train_path, view, "train_data")
    files = os.listdir(path)
    inputs = []
    outs = []
    if view=='human':
        for file in files: 
            # print(os.path.join(path,file))
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = input
            inputs.append(input)
            outs.append(out)        
    else:
        for file in files: 
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = loadmat(os.path.join(path,file))['kSpc']
            inputs.append(input)
            outs.append(out)

    data_in = np.array(inputs)
    data_out = np.array(outs)
    # print(view, "data_in.shape",data_in.shape)
    # transpose shape from [n,96,96,73] -> [n,73,96,96] then reshape to [n*73,96,96]
    data_in    = np.transpose(data_out,(0,3,1,2))
    data_out   = np.transpose(data_out,(0,3,1,2))
    data_c_in  = np.zeros((data_in.shape[0],data_in.shape[1],5,data_in.shape[2],data_in.shape[3]),dtype=np.complex64) # n 
    data_c_out = np.zeros((data_in.shape[0],data_in.shape[1],1,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)

    for i in range(data_in.shape[0]):
        # print("[data_in[i][-1][np.newaxis]", data_in[i][-1][np.newaxis].shape, data_in[i][ :2].shape)
        data_c_in[i][0]  = np.stack([data_in[i][-2],data_in[i][-1], data_in[i][0], data_in[i][1], data_in[i][2]] )
        data_c_in[i][1]  = np.stack([data_in[i][-1],data_in[i][0], data_in[i][1], data_in[i][2], data_in[i][3]] )        
        for j in range(2,data_in.shape[1]-2):
            data_c_in[i][j] = data_in[i][j-2:j+3]

        data_c_in[i][-1] = np.stack([data_in[i][-3], data_in[i][-2], data_in[i][-1], data_in[i][1], data_in[i][0]])
        data_c_in[i][-2] = np.stack([data_in[i][-4], data_in[i][-3], data_in[i][-2], data_in[i][-1] ,data_in[i][0]])

        data_c_out[i][0] = data_out[i][0]
        for j in range(1,data_in.shape[1]-1):
            data_c_out[i][j] = data_out[i][j]
        data_c_out[i][-1] = data_out[i][-1]

    data_c_in = data_c_in.reshape([-1,5,96,96])
    data_out_final = data_c_out.reshape([-1,1,96,96])
    # print("data_in.shape",data_in.shape)
    # print("data_out.shape",data_out.shape)
    # mask = make_mask()
    # masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
    # data_in = masks * data_c_in

    if params.mask_type=="rad":
        mask = loadmat(params.fiverad_path)["d"]
        mask = np.transpose(mask,(2,0,1))
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in
        print("rad sampling")
    if params.mask_type=="loupe":
        mask = np.load(params.loupe_pattern_path)
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in
    if params.mask_type=="full":
        data_in = data_c_in
        mask = np.ones((3,96,96))
    # adjancet_mask = (mask[0] + mask[2] + mask[1]) >=1
    # plt.imshow(adjancet_mask)
    # plt.savefig("three_adjacent.png")
    # masks_sum = (mask[0] + mask[1] + mask[2]) >=1 
    # print("sum_ratio",np.sum(masks_sum)/(masks_sum.shape[1]**2))
    trainx, valx, trainy, valy = train_test_split(data_in, data_out_final, train_size=params.training_percent, random_state=42)
    train_mean = np.mean(trainx)
    train_std = np.std(trainx)
    print("trainx.shape",trainx.shape)
    print("valx.shape",valx.shape)
    return trainx, trainy, valx, valy, train_mean, train_std


def create_test_data(params,test_path,view):
    path = test_path
    files = os.listdir(path)
    inputs = []
    outs = []    
    if view == "human":
        for file in files:
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = input
            inputs.append(input)
            outs.append(out)
    else:
        for file in files:
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = loadmat(os.path.join(path,file))['kSpc']
            inputs.append(input)
            outs.append(out)        
    data_in = np.array(inputs)
    data_out = np.array(outs)
    # transpose shape from [n,96,96,73] -> [n,73,96,96] then reshape to [n*73,96,96]
    data_in = np.transpose(data_in,(0,3,1,2))
    data_out = np.transpose(data_out,(0,3,1,2))
    data_in = data_in.reshape([-1,96,96])
    data_out = data_out.reshape([-1,96,96])
    if params.mask_type=="rad":
        # x4 is radx4 x8 x16 are d 
        mask = loadmat(params.rad_path)["d"]
        masks = np.repeat(mask[np.newaxis],data_in.shape[0],axis=0)
        data_in = masks * data_out
        print("rad sampling")
    return data_in, data_out

def create_test_3chandata(params, test_path,view):
    if os.path.isdir(test_path):
        path = test_path
    else:
        path = [test_path]
    files = os.listdir(path)
    inputs = []
    outs = []
    if view=='human':
        for file in files: 
            # print(os.path.join(path,file))
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = copy.deepcopy(input)
            inputs.append(input)
            outs.append(out)
    else:
        for file in files:
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = loadmat(os.path.join(path,file))['kSpc']
            inputs.append(input)
            outs.append(out)
    print("np.array(inputs).shape",np.array(inputs).shape)
    data_in  = np.transpose(np.array(outs), (0,3,1,2))  #[n, slice_num, 96, 96]
    data_out = np.transpose(np.array(outs), (0,3,1,2))  #[n, slice_num, 96, 96]
    data_c_in =  np.zeros((data_in.shape[0],data_in.shape[1],3,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)
    data_c_out = np.zeros((data_in.shape[0],data_in.shape[1],1,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)

    for i in range(data_in.shape[0]):
        data_c_in[i][0]  = np.stack([data_in[i][-1], data_in[i][0],  data_in[i][1] ] )
        for j in range(1,data_in.shape[1]-1):
            data_c_in[i][j] = data_in[i][j-1:j+2]
        data_c_in[i][-1] = np.stack([data_in[i][-2], data_in[i][-1], data_in[i][0] ])

        data_c_out[i][0] = data_out[i][0]
        for j in range(1,data_in.shape[1]-1):
            data_c_out[i][j] = data_out[i][j]
        data_c_out[i][-1] = data_out[i][-1]

    data_c_in = data_c_in.reshape([-1,3,96,96])
    data_out  = data_c_out.reshape([-1,1,96,96])

    if params.mask_type=="rad":
        mask = loadmat(params.threerad_path)["d"]
        mask = np.transpose(mask,(2,0,1))
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in
        # print("rad sampling")
    if params.mask_type=="rect":
        mask = np.zeros([3,96,96]) # [n w w]
        mask[:,36:60] = 1
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in   
    if params.mask_type=="loupe":
        mask = np.load(params.loupe_pattern_path)
        print("mask.shape",mask.shape)
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in
    return data_in, data_out


def create_test_5chandata(params, test_path,view):
    if os.path.isdir(test_path):
        path = test_path
    else:
        path = [test_path]
    files = os.listdir(path)
    inputs = []
    outs = []
    if view=='human':
        for file in files: 
            # print(os.path.join(path,file))
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = copy.deepcopy(input)
            inputs.append(input)
            outs.append(out)
    else:
        for file in files:
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = loadmat(os.path.join(path,file))['kSpc']
            inputs.append(input)
            outs.append(out)
    print("np.array(inputs).shape",np.array(inputs).shape)
    data_in  = np.transpose(np.array(outs), (0,3,1,2))  # [n,w,h,slice_num] -> [n, slice_num, 96, 96]
    data_out = np.transpose(np.array(outs), (0,3,1,2))  # [n,w,h,slice_num] -> [n, slice_num, 96, 96]
    data_c_in =  np.zeros((data_in.shape[0],data_in.shape[1],5,data_in.shape[2],data_in.shape[3]),dtype=np.complex64) # [n, slice_num ,5, w, h]
    data_c_out = np.zeros((data_in.shape[0],data_in.shape[1],1,data_in.shape[2],data_in.shape[3]),dtype=np.complex64) 

    for i in range(data_in.shape[0]):
        print("[data_in[i][-1][np.newaxis]", data_in[i][-1][np.newaxis].shape, data_in[i][ :2].shape)
        data_c_in[i][0]  = np.stack([data_in[i][-2], data_in[i][-1], data_in[i][0], data_in[i][1], data_in[i][2] ])
        data_c_in[i][1]  = np.stack([data_in[i][-1], data_in[i][0],  data_in[i][1], data_in[i][2], data_in[i][3] ])        
        for j in range(2,data_in.shape[1]-2):
            data_c_in[i][j] = data_in[i][j-2:j+3]
        data_c_in[i][-1] = np.stack([data_in[i][-3], data_in[i][-2], data_in[i][-1], data_in[i][0], data_in[i][1] ])
        data_c_in[i][-2] = np.stack([data_in[i][-4], data_in[i][-3], data_in[i][-2], data_in[i][-1] ,data_in[i][0] ])

        # data_c_out[i][0] = data_out[i][0]
        for j in range(data_in.shape[1]):
            data_c_out[i][j] = data_out[i][j]
        # data_c_out[i][-1] = data_out[i][-1]

    data_c_in = data_c_in.reshape([-1,5,96,96])
    data_out  = data_c_out.reshape([-1,1,96,96])

    if params.mask_type=="rad":
        mask = loadmat(params.fiverad_path)["d"]
        mask = np.transpose(mask,(2,0,1))
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in
        # print("rad sampling")
    if params.mask_type=="loupe":
        mask = np.load(params.loupe_pattern_path)
        print("mask.shape",mask.shape)
        masks = np.repeat(mask[np.newaxis],data_c_in.shape[0],axis=0)
        data_in = masks * data_c_in
    return data_in, data_out



def create_test_3chandata_real(params, test_path,view):
    if os.path.isdir(test_path):
        path = test_path
    else:
        path = [test_path]
    files = os.listdir(path)
    inputs = []
    outs = []
    if params.mask_type=="rad":
        mask = loadmat(params.threerad_path)["d"]
        mask = np.transpose(mask,(2,0,1))
        print("rad sampling","mask.shape",mask.shape)
    if params.mask_type=="loupe":
        mask = np.load(params.loupe_pattern_path)
        print("mask.shape",mask.shape)
    masks = np.repeat(mask,32,axis=0)

    if view=='human':
        for file in files: 
            # print(os.path.join(path,file))
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = copy.deepcopy(input)
            inputs.append(out*masks)
            outs.append(out)
    else:
        for file in files:
            input = loadmat(os.path.join(path,file))[params.acc_name]
            out = loadmat(os.path.join(path,file))['kSpc'] #fully sampled
            print("out.shape",out.shape)
            inputs.append(out*masks)
            outs.append(out)
    print("np.array(inputs).shape",np.array(inputs).shape)

    data_in  = np.transpose(np.array(outs), (0,3,1,2))  #[n, slice_num, 96, 96]
    data_out = np.transpose(np.array(outs), (0,3,1,2))  #[n, slice_num, 96, 96]
    data_c_in =  np.zeros((data_in.shape[0],data_in.shape[1],3,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)
    data_c_out = np.zeros((data_in.shape[0],data_in.shape[1],1,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)

    for i in range(data_in.shape[0]):
        data_c_in[i][0]  = np.stack([data_in[i][-1], data_in[i][0],  data_in[i][1] ] )
        for j in range(1,data_in.shape[1]-1):
            data_c_in[i][j] = data_in[i][j-1:j+2]
        data_c_in[i][-1] = np.stack([data_in[i][-2], data_in[i][-1], data_in[i][0] ])

        data_c_out[i][0] = data_out[i][0]
        for j in range(1,data_in.shape[1]-1):
            data_c_out[i][j] = data_out[i][j]
        data_c_out[i][-1] = data_out[i][-1]

    data_c_in = data_c_in.reshape([-1,3,96,96])
    data_out  = data_c_out.reshape([-1,1,96,96])

    return data_in, data_out




def adjust_learning_rate(epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs
    Parameters
    ----------
    epoch : int
        The epoch number.
    optimizer : torch.optim.Optimizer
        The optimizer.

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The optimizer.
    """
    lr = params.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


#math ================================
def c2r(complex_img, axis=0):
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.stack((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(real_img, axis=0):
    """
    :input shape: 2 x row x col (float32)
    :output shape: row x col (complex64)
    """
    if axis == 0:
        complex_img = real_img[0] + 1j*real_img[1]
    elif axis == 1:
        complex_img = real_img[:,0] + 1j*real_img[:,1]
    else:
        raise NotImplementedError
    return complex_img

def make_mask():
    if params.mask_type == "rad" and params.n_channels == 1:
        mask = loadmat(params.threerad_path)["d"]
        masks = np.transpose(mask,(2,0,1))[1]
        print("rad sampling 1",masks.shape,np.sum(masks)/(96*96))
        masks = np.repeat(masks[np.newaxis],999,axis=0)
        masks = torch.from_numpy(masks)
    if params.mask_type == "rad" and params.n_channels == 3:
        mask = loadmat(params.threerad_path)["d"]
        masks = np.transpose(mask,(2,0,1))[1]
        joint_mask = mask[:,:,0] + mask[:,:,1] + mask[:,:,2]
        joint_mask[joint_mask >= 1] = 1
        print("rad sampling 3",masks.shape,np.sum(masks)/(96*96), joint_mask.shape,np.sum(joint_mask)/(96*96))
        masks = np.repeat(masks[np.newaxis],999,axis=0)
        masks = torch.from_numpy(masks)
    if params.mask_type == "rad" and params.n_channels == 5:
        mask = loadmat(params.fiverad_path)["d"]
        masks = np.transpose(mask,(2,0,1))[2]
        joint_mask = mask[:,:,0] + mask[:,:,1] + mask[:,:,2] + mask[:,:,3] + mask[:,:,4]
        joint_mask[joint_mask >= 1] = 1
        print("rad sampling 5",masks.shape,np.sum(masks)/(96*96), joint_mask.shape,np.sum(joint_mask)/(96*96))
        masks = np.repeat(masks[np.newaxis],999,axis=0)
        masks = torch.from_numpy(masks)
    if params.mask_type == 'rect':
        masks = torch.zeros([999,96,96],dtype=torch.int32 ) # [n w w]
        masks[:,36:60] = 1
    if params.mask_type == "loupe":
        masks = np.load(params.loupe_pattern_path)[0][1]
        masks = np.repeat(masks[np.newaxis],999,axis=0)
        masks = torch.from_numpy(masks)
    # print("np.sum(masks[0,:,:,:])",np.sum(masks[0,:,:,:]))
    # print("masks.shape",masks.shape)
    return masks