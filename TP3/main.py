from datasets import load_dataset
from base64 import b64decode
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    print(fps, num_frames)

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    # plt.show() 
    while True:
        _, image = vidcap.read()
        
        try:
            red = cv2.calcHist([image], [2], None, [64], [0, 256])
            green = cv2.calcHist([image], [1], None, [64], [0, 256])
            blue = cv2.calcHist([image], [0], None, [64], [0, 256])
            hist = np.concatenate((red, green, blue), axis=0)
        except:
            break 
        
        hist_matrix.append(hist)
        index.append([video_path[-8:-4], round(vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 6)])

    print(len(hist_matrix), len(hist_matrix[0]))
    # plt.plot(hist_matrix[0,0:64], color='b') 
    # plt.title('Image Histogram For Blue Channel GFG') 
    # plt.show()


def QUESTION1():
    # data = load_dataset('pinecone/image-set', split='train')
    # images = [process_fn(sample) for sample in data]
    for i in range(1, 3):
        index_build(f'data/mp4/v00{i}.mp4')







if __name__ == '__main__':
    QUESTION1()