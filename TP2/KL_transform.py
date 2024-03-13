import numpy as np
import matplotlib.pyplot as py
from numpy import linalg as LA
import cv2

height = 0
width = 0

def kl_transform(image: str, level: tuple[int,int,int], color_space: str, show_img = False, isq4 = False, imageQ4 = None):
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

    eigvec = np.transpose(eigvec)
    vecMoy = np.array([[MoyR], [MoyG], [MoyB]]).reshape(1,3) 

    image_flat = image.reshape(height * width, 3)
    diff = image_flat - vecMoy


    if isq4:
        quantification(diff, level)
        imagelue = cv2.imread(imageQ4)
        print(imageQ4)
        if color_space == 'YUV':
            image = cv2.cvtColor(imagelue, cv2.COLOR_BGR2YUV)
        elif color_space == 'RGB':
            image = cv2.cvtColor(imagelue, cv2.COLOR_BGR2RGB)
        image = np.array(image.astype('double'))

        MoyR = np.mean(image[:, :, 0])
        MoyG = np.mean(image[:, :, 1])
        MoyB = np.mean(image[:, :, 2])

        image_flat = image.reshape(height * width, 3)
    else:
        pass


    imageKL_flat = np.dot(eigvec, diff.T).T
    imageKL_flat = quantification(imageKL_flat, level)
    invEigvec = LA.pinv(eigvec);

    vecMoy = [MoyR, MoyG, MoyB]

    imageRGB_flat = np.dot(invEigvec, imageKL_flat.T).T + vecMoy
    imageRGB = imageRGB_flat.reshape(height, width, 3)

    if color_space == 'YUV':
        KLimage = np.clip(imageRGB, 0, 128).astype('uint8')
    else:
        KLimage = np.clip(imageRGB,0,255).astype('uint8')

    if show_img:
        py.figure(figsize = (10,10))
        py.imshow(KLimage)
        py.show()

    return KLimage

def quantification(image, levels):
    rounded_arr = np.zeros(image.shape)

    for i, l in enumerate(levels):
        if l == 0:
            rounded_arr[:, i] = np.zeros(image[:, i].shape)
            continue

        min_value = np.min(image[:, i])
        max_value = np.max(image[:, i])

        channel = image[:, i]

        levels_values = np.linspace(min_value, max_value, num=2**l)

        nearest_vals_indices = np.abs(levels_values - channel[:, np.newaxis]).argmin(axis=1)
        rounded_arr[:, i] = levels_values[nearest_vals_indices]

    return rounded_arr
