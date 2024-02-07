import numpy as np
import matplotlib.pyplot as py
from numpy import linalg as LA
import cv2

height = 0
width = 0

def kl_transform(image: str, show_img = False, color_space: str = 'RGB'):
    global height, width
    imagelue = cv2.imread(image)

    if color_space == 'YUV':
        image = cv2.cvtColor(imagelue, cv2.COLOR_BGR2YUV)
    elif color_space == 'RGB':
        image = cv2.cvtColor(imagelue, cv2.COLOR_BGR2RGB)

    if show_img and color_space == 'RGB':
        py.figure(figsize = (10,10))
        py.imshow(image)
        py.show()

    image= np.array(image.astype('double'))
    height = len(image)
    width = len(image[0])

    nbPixels = len(image)*len(image[0])        
    MoyR = np.mean(image[:, :, 0])
    MoyG = np.mean(image[:, :, 1])
    MoyB = np.mean(image[:, :, 2])

    # print(MoyR)
    # print(MoyG)
    # print(MoyB)

    diff_image = image - [MoyR, MoyG, MoyB]
    reshaped_diff_image = diff_image.reshape(-1, 3)
    covRGB = np.dot(reshaped_diff_image.T, reshaped_diff_image)

    covRGB = covRGB / nbPixels        
    # print(covRGB)

    eigval, eigvec = LA.eig(covRGB)
    # print(eigval)
    # print()
    # print(eigvec)

    eigvec = np.transpose(eigvec)
    eigvecsansAxe0 = np.copy(eigvec)
    eigvecsansAxe0[0,:] = [0.0,0.0,0.0]
    eigvecsansAxe1 = np.copy(eigvec)
    eigvecsansAxe1[1,:] = [0.0,0.0,0.0]
    eigvecsansAxe2 = np.copy(eigvec)
    eigvecsansAxe2[2,:] = [0.0,0.0,0.0]

    vecMoy = np.array([[MoyR], [MoyG], [MoyB]]).reshape(1,3) 

    image_flat = image.reshape(height * width, 3)

    diff = image_flat - vecMoy

    imageKLsansAxe0_flat = np.dot(eigvecsansAxe0, diff.T).T
    imageKLsansAxe1_flat = np.dot(eigvecsansAxe1, diff.T).T
    imageKLsansAxe2_flat = np.dot(eigvecsansAxe2, diff.T).T

    print(imageKLsansAxe2_flat)

    imageKLsansAxe0_flat = quantification(imageKLsansAxe0_flat)
    imageKLsansAxe1_flat = quantification(imageKLsansAxe1_flat)
    imageKLsansAxe2_flat = quantification(imageKLsansAxe2_flat)

    invEigvecsansAxe0 = LA.pinv(eigvecsansAxe0);
    invEigvecsansAxe1 = LA.pinv(eigvecsansAxe1);
    invEigvecsansAxe2 = LA.pinv(eigvecsansAxe2);

    vecMoy =[MoyR, MoyG, MoyB] 

    imageRGBsansAxe0_flat = np.dot(invEigvecsansAxe0, imageKLsansAxe0_flat.T).T + vecMoy
    imageRGBsansAxe1_flat = np.dot(invEigvecsansAxe1, imageKLsansAxe1_flat.T).T + vecMoy
    imageRGBsansAxe2_flat = np.dot(invEigvecsansAxe2, imageKLsansAxe2_flat.T).T + vecMoy

    imageRGBsansAxe0 = imageRGBsansAxe0_flat.reshape(height, width, 3)
    imageRGBsansAxe1 = imageRGBsansAxe1_flat.reshape(height, width, 3)
    imageRGBsansAxe2 = imageRGBsansAxe2_flat.reshape(height, width, 3)

    if color_space == 'YUV':
        KLimage0 = np.clip(imageRGBsansAxe0, 0, 128).astype('uint8')
        KLimage1 = np.clip(imageRGBsansAxe1, 0, 128).astype('uint8')
        KLimage2 = np.clip(imageRGBsansAxe2, 0, 128).astype('uint8')     
    else:
        KLimage0 = np.clip(imageRGBsansAxe0,0,255).astype('uint8')
        KLimage1 = np.clip(imageRGBsansAxe1,0,255).astype('uint8')
        KLimage2 = np.clip(imageRGBsansAxe2,0,255).astype('uint8')

    if show_img:
        for imageout in [KLimage0, KLimage1, KLimage2]:
            py.figure(figsize = (10,10))
            py.imshow(imageout)
            py.show()

    return KLimage0, KLimage1, KLimage2


def quantification(image, levels=[8, 8, 8]):
    min_value = np.min(image)
    max_value = np.max(image)

    rounded_arr = [] 
    levels_values = np.linspace(min_value, max_value, num=2**8)

    image = image.flatten()

    for num in image:
        nearest_val = min(levels_values, key=lambda x: abs(x - num))
        rounded_arr.append(nearest_val)
    return np.array(rounded_arr).reshape(height * width, 3)
