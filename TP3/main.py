# from datasets import load_dataset
from base64 import b64decode
from cv2.typing import MatLike
import cv2, os, csv
import numpy as np
import matplotlib.pyplot as plt
# import torchvision
import pickle, time
import scipy.spatial.distance as dist
BIN = 12

# F1 score https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
index:list[tuple[str, float]] = []
hist_matrix: list[MatLike] = []

IMG_FOLDER = 'data/jpeg'
VIDEO_FOLDER = 'data/mp4'

F1_SCORE = 0.90

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
    indexed_frames = 0
    while True:
        _, image = vidcap.read()
        #if (count % 10) == 0:
        try:
                hist = image_hist(image)
                indexed_frames += 1
        except:
            break 
        
        hist_matrix.append(hist)
        index.append([video_path[-8:-4], round(vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 6)])
        count += 1
    # indx = 10
    # while indx < num_frames:
    #     vidcap.set(cv2.CAP_PROP_POS_FRAMES, indx)
    #     _, image = vidcap.read()
    #     hist = image_hist(image)
    #     hist_matrix.append(hist)
    #     index.append([video_path[-8:-4], round(vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 6)])
    #     indx += 10
    #     count += 1
    print(f'Indexed {video_path} with {count} frames')
    # print(f'Hist shape {hist.shape}')
    # plt.plot(hist_matrix[0,0:64], color='b') 
    # plt.title('Image Histogram For Blue Channel GFG') 
    # plt.show()


def image_hist(image):
    red = cv2.calcHist([image], [2], None, [BIN], [0, 255])
    green = cv2.calcHist([image], [1], None, [BIN], [0, 255])
    blue = cv2.calcHist([image], [0], None, [BIN], [0, 255])
    hist = np.concatenate((red, green, blue), axis=0)

    # hist = cv2.calcHist([image], [0,1,2], None, [BIN, BIN, BIN], [0, 255, 0, 255, 0, 255])
    cv2.normalize(hist, hist)
    return hist.flatten()

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_cosine(image_to_find: np.ndarray, top_k=1):
    global hist_matrix
    
    distances = []
    for i, vector in enumerate(hist_matrix):
        similarity = cosine(image_to_find, np.array(vector))
        distances.append((i, similarity))  # Store index and similarity

    distances.sort(key=lambda x: x[1], reverse=True)
    print(distances[:1])

    top_indices = [index if distance > F1_SCORE else 'out' for index, distance in distances[:top_k]]
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

def save_vars(range_end=100):
    global index, hist_matrix
    start_1 = time.time()
    for i in range(1, range_end):
        index_build(f'{VIDEO_FOLDER}/v{i:03d}.mp4')
    print(f"Indexation time: {time.time() - start_1} seconds")
    print('matrix size in bytes:', len(pickle.dumps(hist_matrix)))

    with open('data/index.pkl', 'wb') as save_index:
        pickle.dump(index, save_index)

    with open('data/hist_matrix.pkl', 'wb') as save_hist:
        pickle.dump(hist_matrix, save_hist)

def load_vars():
    global index, hist_matrix
    with open('data/index.pkl', 'rb') as load_index:
        index = pickle.load(load_index)

    with open('data/hist_matrix.pkl', 'rb') as load_hist:
        hist_matrix = pickle.load(load_hist)

def evaluate_result(exepected, actual):
    if actual == 'out':
        if exepected == actual:
            return 'TN'
        else:
            return 'FN'
    else:
        if exepected == actual:
            return 'TP'
        else:
            return 'FP'
        
def time_delta(time1, time2):
    return abs(time1 - time2)

def QUESTION1(result_csv: csv.writer, folder=IMG_FOLDER):
    global index, hist_matrix
    #save_vars(101)
    load_vars()
    # print(index[0:5])
    # print(hist_matrix[0:5])

    print(f"nombre de frame dans l'index {len(index)}")
    times = []

    with open('data/gt.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header

        for i, (row, filename) in enumerate(zip(reader, os.listdir(folder))):
            # print(row,i, filename)
            # if i > 10: 
            #     break
            if filename.endswith(".jpeg"):
                start_2 = time.time()
                image = cv2.imread(os.path.join(folder, filename))
                vector = image_hist(image)
                #id = search_cosine(vector, 3)
                id = search_euclidean(vector, 3)
                # print(id) 
                times.append(time.time() - start_2)
                #print(f"Image {filename} processed in {time.time() - start_2} seconds")

                if id[0] != 'out':

                    print(f'Image: {filename} is similar to vidéo: {index[id[0]][0]} at {index[id[0]][1]}s that is {evaluate_result(row[1], index[id[0]][0])}')
                    # result_csv.writerow([filename, index[id[0]][0], time_delta(float(index[id[0]][1]), float(row[2])) if row[2] else '0', evaluate_result(row[1], index[id[0]][0])])
                    result_csv.writerow([filename.replace('.jpeg', ''), index[id[0]][0], float(index[id[0]][1]) if row[2] else '0', evaluate_result(row[1], index[id[0]][0])])
                else:
                    print(f"Image: {filename} is not similar to any video {evaluate_result(row[1], 'out')}")
                    result_csv.writerow([filename.replace('.jpeg', ''), 'out', '', evaluate_result(row[1], 'out')])
    
    print(f"Average time per image: {sum(times) / len(times)} seconds")

if __name__ == '__main__':
    print(BIN)
    start = time.time()
    with open('result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'video_pred', 'minutage_pred', "evaluation"]) # Evaluation is TP, FP, TN, FN
        QUESTION1(writer)
    print(f"Execution time: {time.time() - start} seconds")

#  Bin evaluation               F1 score eval
#  Bin = 256 -> 70.1% TP
#  Bin = 85 -> 73.9% TP
#  Bin = 4 -> 76.3% TP
#  Bin = 8 -> 78.3% TP
    

#  Bin = 12 -> 78.5% TP    --> 0.9 : 85%
    


#  Bin = 16 -> 77.9% TP
#  Bin = 32 -> 76.4% TP
#  Bin = 64 -> 
    
# Goal: 80.5% 