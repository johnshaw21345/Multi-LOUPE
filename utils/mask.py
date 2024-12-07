import numpy as np


def fixed_cartesian_mask_3ch(sample_rate, img_size=200):
    mask_cartesian = generate_cartesian_sampling_mask(img_size=img_size, sample_percentage=sample_rate)
    masks = np.stack((mask_cartesian,mask_cartesian,mask_cartesian))
    return masks

def random_cartesian_mask_3ch(sample_rate, img_size=200):
    cf = 0.1
    masks = np.stack((variable_density_cartesian_sampling(img_shape = (img_size,img_size), center_fraction=cf, center_sampling_rate=sample_rate*4, outer_sampling_rate=sample_rate/(1-cf)*(1-cf*4)),
                        variable_density_cartesian_sampling(img_shape = (img_size,img_size), center_fraction=cf, center_sampling_rate=sample_rate*4, outer_sampling_rate=sample_rate/(1-cf)*(1-cf*4)),
                        variable_density_cartesian_sampling(img_shape = (img_size,img_size), center_fraction=cf, center_sampling_rate=sample_rate*4, outer_sampling_rate=sample_rate/(1-cf)*(1-cf*4))))
    
    return masks

def fixed_radial_mask_3ch(num_spokes, img_size=200):
    mask_radial = generate_radial_sampling_mask(img_size=img_size, num_spokes=num_spokes)
    masks = np.stack((mask_radial,mask_radial,mask_radial))
    return masks

def rotated_radial_mask_3ch(num_spokes, img_size=200):
    mask_radial_0 = generate_radial_sampling_mask(img_size=img_size, num_spokes=28, bias=np.pi/3/num_spokes)
    mask_radial_1 = generate_radial_sampling_mask(img_size=img_size, num_spokes=28)
    mask_radial_2 = generate_radial_sampling_mask(img_size=img_size, num_spokes=28, bias=-np.pi/3/num_spokes)
    masks = np.stack((mask_radial_0,mask_radial_1,mask_radial_2))
    return masks

def fixed_cartesian_mask_1ch(sample_rate, img_size=200):
    mask_cartesian = generate_cartesian_sampling_mask(img_size=img_size, sample_percentage=sample_rate)
    masks = mask_cartesian[None,...]
    return masks

def fixed_radial_mask_1ch(num_spokes, img_size=200):
    mask_radial = generate_radial_sampling_mask(img_size=img_size, num_spokes=num_spokes)
    masks = mask_radial[None,...]
    return masks

def generate_cartesian_sampling_mask(img_size=512, sample_percentage=0.25):
    mask = np.zeros((img_size,img_size))
    s = np.linspace(0,1,int(img_size/2*sample_percentage))
    y=(int(img_size/2-1)*s*s).astype(int)
    mask[int(img_size/2)+y]=1
    mask[int(img_size/2-1)-y]=1
    return mask

def generate_radial_sampling_mask(img_size=512, num_spokes=16, bias=0): 
    mask = np.zeros((img_size,img_size))
    center = img_size//2
    angles = np.linspace(0,np.pi,num_spokes) + bias

    for angle in angles:
        for r in range(-int(0.5*img_size), int(0.5*img_size)+1):
            x = int(center + r * np.cos(angle))
            y = int(center + r * np.sin(angle))
            if 0<x<img_size and 0<y<img_size:
                mask[x][y]=1
    
    return mask

def variable_density_cartesian_sampling(img_shape, center_fraction, center_sampling_rate, outer_sampling_rate): 
    """ Generate a Cartesian sampling pattern with variable density for k-space. Parameters: img_shape (tuple): The shape of the image (height, width). center_fraction (float): Fraction of k-space (height) considered as center region. center_sampling_rate (float): Sampling rate in the center region (0 < center_sampling_rate <= 1). outer_sampling_rate (float): Sampling rate in the high-frequency region (0 < outer_sampling_rate <= 1). Returns: np.ndarray: A binary sampling mask of the same size as img_shape. """ 
    # Initialize mask with zeros 
    mask = np.zeros(img_shape) 
    # Determine center region limits 
    center_size = int(img_shape[0] * center_fraction) 
    center_start = (img_shape[0] - center_size) // 2 
    center_end = center_start + center_size 
    # Sample center region with center_sampling_rate 
    center_lines = np.random.choice( range(center_start, center_end), size=int(center_size * center_sampling_rate), replace=False ) 
    # Set selected lines to 1 in the center region 
    mask[center_lines, :] = 1 
    
    # Sample outer (high-frequency) region with outer_sampling_rate 
    outer_lines = np.random.choice( np.concatenate((range(0, center_start), range(center_end, img_shape[0]))), size=int((img_shape[0] - center_size) * outer_sampling_rate), replace=False ) 
    mask[outer_lines, :] = 1 
    # Set selected lines to 1 in the outer region 
    return mask 


