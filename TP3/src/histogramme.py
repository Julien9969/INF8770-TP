import base64
import cv2
import numpy as np
from datasets import load_dataset  
from io import BytesIO
import os
import matplotlib.pyplot as plt


data = load_dataset('TP3/data/jpeg', split='train', revision='e7d39fc')
folder = 'TP3/data/jpeg'

images = []
def process_fn():
    for filename in os.listdir(folder):
        if filename.endswith(".jpeg"):
            with open(os.path.join(folder, filename), 'rb') as f:
                image_bytes = f.read()
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                #convert from BGR to RGB
                rgb_image = np.flip(image, 2)
                images.append(rgb_image)
# Convert the image to RGB
# image = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
# print(image.shape)
# plt.imshow(image)
# rgb_image = np.flip(images[0], 2)
# shape = rgb_image.shape
# shape, rgb_image[0, 0, :]

# def build_histogram(image, bins=256):
#     # convert from BGR to RGB
#     rgb_image = np.flip(image, 2)
#     # show the image
#     plt.imshow(rgb_image)
#     # convert to a vector
#     image_vector = rgb_image.reshape(1, -1, 3)
#     # break into given number of bins
#     div = 256 / bins
#     bins_vector = (image_vector / div).astype(int)
#     # get the red, green, and blue channels
#     red = bins_vector[0, :, 0]
#     green = bins_vector[0, :, 1]
#     blue = bins_vector[0, :, 2]
#     # build the histograms and display
#     fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
#     axs[0].hist(red, bins=bins, color='r')
#     axs[1].hist(green, bins=bins, color='g')
#     axs[2].hist(blue, bins=bins, color='b')
#     plt.show()

# build_histogram(images[6], 64)
process_fn()
def build_histogram(img_indx=3, bins=64):
    plt.imshow(images[0])
    red_hist = cv2.calcHist(
        [images[img_indx]], [2], None, [bins], [0, 256]
    )
    green_hist = cv2.calcHist(
        [images[img_indx]], [1], None, [bins], [0, 256]
    )
    blue_hist = cv2.calcHist(
        [images[img_indx]], [0], None, [bins], [0, 256]
    )    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    axs[0].plot(red_hist, color='r')
    axs[1].plot(green_hist, color='g')
    axs[2].plot(blue_hist, color='b')
    plt.show()

# build_histogram()



def get_vector(image, bins=32):
    red = cv2.calcHist(
        [image], [2], None, [bins], [0, 256]
    )
    green = cv2.calcHist(
        [image], [1], None, [bins], [0, 256]
    )
    blue = cv2.calcHist(
        [image], [0], None, [bins], [0, 256]
    )
    vector = np.concatenate([red, green, blue], axis=0)
    vector = vector.reshape(-1)
    return vector

# and do the same for the remainder of our images
image_vectors = []
for image in images:
    image_vectors.append(get_vector(image))


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# search for the most similar images to the first image 
def search(idx, top_k=5):
    query_vector = image_vectors[idx]
    distances = []
    for _, vector in enumerate(image_vectors):
        distances.append(cosine(query_vector, vector))
    # get top k most similar images
    top_idx = np.argpartition(distances, -top_k)[-top_k:]
    return top_idx

#plot the most similar images
def plot_similar_images(idx, top_k=5):
    similar_images = search(idx, top_k)
    fig, axs = plt.subplots(1, top_k, figsize=(15, 4), sharey=True)
    for i, ax in enumerate(axs):
        ax.imshow(images[similar_images[i]])
    plt.show()

plot_similar_images(0, 5)