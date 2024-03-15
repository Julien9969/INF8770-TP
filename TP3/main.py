from datasets import load_dataset
from base64 import b64decode
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
# import torchvision
import torch
import scipy.spatial.distance as dist
BIN = 128
# BIN = 150

# F1 score https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
index = []
hist_matrix = []

def process_fn(sample):
    image_bytes = b64decode(sample['image_bytes'])
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return image


def index_build(video_path = 'data/mp4/v001.mp4'):
    global index, hist_matrix

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    # plt.show() 
    count = 1

    while True:
        _, image = vidcap.read()
        
        try:
            hist = image_hist(image)
        except:
            break 
        
        hist_matrix.append(hist)
        index.append([video_path[-8:-4], round(vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 6)])
        count += 1

    print(f'Indexed {video_path} with {num_frames} frames')
    # print(f'Hist shape {hist.shape}')
    # plt.plot(hist_matrix[0,0:64], color='b') 
    # plt.title('Image Histogram For Blue Channel GFG') 
    # plt.show()


def image_hist(image):
    red = cv2.calcHist([image], [2], None, [BIN], [0, 256])
    green = cv2.calcHist([image], [1], None, [BIN], [0, 256])
    blue = cv2.calcHist([image], [0], None, [BIN], [0, 256])
    hist = np.concatenate((red, green, blue), axis=0)

    # hist = cv2.calcHist([image], [0,1,2], None, [BIN, BIN, BIN], [0, 256, 0, 256, 0, 256])

    cv2.normalize(hist, hist)
    return hist.flatten()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def search(image_to_find: np.ndarray, top_k=1):
#     distances = []
#     for i, vector in enumerate(hist_matrix):
#         # distances.append(dist.euclidean(image_to_find.flatten(), vector.flatten()))
#         distances.append(cosine(image_to_find, np.array(vector)))
#         # distances.append(dist.cosine(image_to_find, vector))
#     # get top k most similar images
#     top_idx = np.argpartition(distances, -top_k)[-top_k:]
#     return top_idx

def search_cosine(image_to_find: np.ndarray, top_k=1):
    global hist_matrix
    
    distances = []
    for i, vector in enumerate(hist_matrix):
        similarity = cosine(image_to_find, np.array(vector))
        distances.append((i, similarity))  # Store index and similarity

    distances.sort(key=lambda x: x[1], reverse=True)

    top_indices = [index for index, _ in distances[:top_k]]
    return top_indices

def search_euclidean(image_to_find: np.ndarray, top_k=1):
    global hist_matrix
    
    distances = []
    for i, vector in enumerate(hist_matrix):
        distance = np.linalg.norm(image_to_find - np.array(vector))
        distances.append((i, distance))  # Store index and distance

    distances.sort(key=lambda x: x[1])

    top_indices = [index for index, _ in distances[:top_k]]
    return top_indices

def QUESTION1(folder = 'data/jpeg'):
    # data = load_dataset('pinecone/image-set', split='train')
    # images = [process_fn(sample) for sample in data]
    for i in range(1, 77):
        index_build(f'data/mp4/v{i:03d}.mp4')

    # index_build(f'data/mp4/v00{1}.mp4')
    # index_build(f'data/mp4/v0{50}.mp4')
    # index_build(f'data/mp4/v0{42}.mp4')


    global index, hist_matrix
    print(f"nombre de frame dans l'index {len(index)}")

    # hist_matrix = np.array(hist_matrix)

    for i, filename in enumerate(os.listdir(folder)):

        if i > 5:
            break

        if filename.endswith(".jpeg"):
            image = cv2.imread(os.path.join(folder, filename))

            # plt.imshow(image.)

            # rgb_image = np.flip(image, 2)
            vector = image_hist(image)
            id = search_cosine(vector, 3)
            # id = search_euclidean(vector, 3)
            # print(id)
            print(f'Image: {filename} is similar to frames: {index[id[0]]} and {index[id[1]]} and {index[id[2]]}')








if __name__ == '__main__':
    QUESTION1()