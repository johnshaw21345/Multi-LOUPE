import os
import torch
import numpy as np
from torch.utils.data import Dataset


def create_my_data_3ch(params):
    """Create data basedset for training and validation.
    Note that in practice, at test time the method will need to be applied to
    the whole volume. In addition, one would need more data to prevent
    overfitting.
    """
    path = os.path.join(params.train_path, "train_data")
    files = os.listdir(path)
    inputs = []
    outs = []    
     
    for file in files: 
        input = np.load(os.path.join(path,file))
        out = input
        inputs.append(input)
        outs.append(out)       

    data_in = np.array(inputs)
    data_out = np.array(outs)

    # # transpose shape from [n,8,200,200] -> [n,8,200,200] then reshape to [n*8,200,200]
    data_c_in  = np.zeros((data_in.shape[0],data_in.shape[1],3,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)
    data_c_out = np.zeros((data_in.shape[0],data_in.shape[1],3,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)

    for i in range(data_in.shape[0]):
        # print("[data_in[i][-1][np.newaxis]", data_in[i][-1][np.newaxis].shape, data_in[i][ :2].shape)
        data_c_in[i][0]  = np.stack([data_in[i][-1], data_in[i][0], data_in[i][1]])
        for j in range(1,data_in.shape[1]-1):
            data_c_in[i][j] = data_in[i][j-1:j+2]
        data_c_in[i][-1] = np.stack([data_in[i][-2], data_in[i][-1], data_in[i][0]])

        data_c_out[i][0] = np.stack([data_out[i][-1], data_out[i][0], data_out[i][1]])
        for j in range(1,data_in.shape[1]-1):
            data_c_out[i][j] = data_out[i][j-1:j+2]
        data_c_out[i][-1] = np.stack([data_out[i][-2], data_out[i][-1], data_out[i][0]])

    print(data_c_in.shape)
    print(data_c_out.shape)

    data_c_in = data_c_in.reshape([-1,3,200,200])
    data_out_final = data_c_out.reshape([-1,3,200,200])

    trainx, valx, trainy, valy = train_test_split(data_c_in, data_out_final, train_size=params.training_percent, random_state=42)
    train_mean = np.mean(trainx)
    train_std = np.std(trainx)
    print("trainx.shape",trainx.shape)
    print("valx.shape",valx.shape)
    return trainx, trainy, valx, valy, train_mean, train_std

def create_my_data_1ch(params):
    """Create data basedset for training and validation.
    Note that in practice, at test time the method will need to be applied to
    the whole volume. In addition, one would need more data to prevent
    overfitting.
    """
    path = os.path.join(params.train_path, "train_data")
    files = os.listdir(path)
    inputs = []
    outs = []    
     
    for file in files: 
        input = np.load(os.path.join(path,file))
        out = input
        inputs.append(input)
        outs.append(out)       

    data_in = np.array(inputs)
    data_out = np.array(outs)

    # # transpose shape from [n,8,200,200] -> [n,8,200,200] then reshape to [n*8,200,200]
    data_c_in  = np.zeros((data_in.shape[0],data_in.shape[1],1,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)
    data_c_out = np.zeros((data_in.shape[0],data_in.shape[1],1,data_in.shape[2],data_in.shape[3]),dtype=np.complex64)

    for i in range(data_in.shape[0]):
        for j in range(data_in.shape[1]):
            data_c_in[i][j] = data_in[i][j]
            data_c_out[i][j] = data_out[i][j]

    print(data_c_in.shape)
    print(data_c_out.shape)

    data_c_in = data_c_in.reshape([-1,1,200,200])
    data_out_final = data_c_out.reshape([-1,1,200,200])

    trainx, valx, trainy, valy = train_test_split(data_c_in, data_out_final, train_size=params.training_percent, random_state=42)
    train_mean = np.mean(trainx)
    train_std = np.std(trainx)
    print("trainx.shape",trainx.shape)
    print("valx.shape",valx.shape)
    return trainx, trainy, valx, valy, train_mean, train_std

class BasicDataset(Dataset):
    def __init__(self, imgs, masks):
        self.k_imgs_un = torch.from_numpy(imgs)
        self.k_imgs_full = torch.from_numpy(masks)
        self.imgs_un = torch.fft.ifft2(self.k_imgs_un)
        self.imgs_full = torch.fft.ifft2(self.k_imgs_full)
        print("self.imgs_full.shape",self.imgs_full.shape, "self.imgs_un.shape",self.imgs_un.shape,"self.k_imgs_full.shape",self.k_imgs_full.shape)
        self.w , self.h = self.imgs_full.shape[-1],self.imgs_full.shape[-2]

    def __len__(self):
        return len(self.k_imgs_un)

    def __getitem__(self, i):
        '''
        c_last:[n 1 w h 2] 
        c_sec:[n 2 w h]
        '''

        img_full = self.imgs_full[i]
        img_full_real = torch.abs(img_full)
        img_un = self.imgs_un[i]

        # print("img_un.shape",img_un.shape)            
        img_un = torch.cat((img_un.real,img_un.imag),dim=0)#.reshape([-1,self.w,self.h])#[0:2]
        return {'img_un':img_un,"img_full_real":img_full_real}
     
        # k_full =torch.squeeze( torch.stack((k_img_full.real,k_img_full.imag),axis=0))
        # img_full =torch.squeeze( torch.stack((img_full.real,img_full.imag),axis=0))

from sklearn.model_selection import train_test_split
