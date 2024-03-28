import cv2, os, csv, pickle, time
from cv2.typing import MatLike
import numpy as np

index:list[tuple[str, float]] = []
hist_matrix: list[MatLike] = []

F1_SCORE_HIST = 0.91
BIN = 12
IMG_FOLDER = 'data/jpeg'
VIDEO_FOLDER = 'data/mp4'

def save_vars(range_end=100):
    global index, hist_matrix

    for i in range(1, range_end):
        index_build_hist(f'{VIDEO_FOLDER}/v{i:03d}.mp4')

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

def image_hist(image):
    red = cv2.calcHist([image], [2], None, [BIN], [0, 255])
    green = cv2.calcHist([image], [1], None, [BIN], [0, 255])
    blue = cv2.calcHist([image], [0], None, [BIN], [0, 255])
    hist = np.concatenate((red, green, blue), axis=0)

    cv2.normalize(hist, hist)
    return hist.flatten()

def index_build_hist(video_path = 'data/mp4/v001.mp4'):
    global index, hist_matrix

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
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

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_cosine(image_to_find, top_k=1):
    global hist_matrix
    
    distances = []
    for i, vector in enumerate(hist_matrix):
        similarity = cosine(image_to_find, np.array(vector))
        distances.append((i, similarity))  # Store index and similarity

    distances.sort(key=lambda x: x[1], reverse=True)

    top_indices = [index if distance > F1_SCORE_HIST else 'out' for index, distance in distances[:top_k]]
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
        
def hist_find(result_csv: csv.writer, folder=IMG_FOLDER):
    global index, hist_matrix
    index_time = None
    try:
        load_vars()
        index_time = 'loaded from file'
        pass
    except FileNotFoundError:
        start = time.time()
        save_vars(101)
        index_time = time.time() - start
    # load_vars()

    print(f"nombre de frame dans l'index {len(index)}")

    with open('data/gt.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header
        start = time.time()
        for i, (row, filename) in enumerate(zip(reader, os.listdir(folder))):
            if filename.endswith(".jpeg"):
                image = cv2.imread(os.path.join(folder, filename))


                vector = image_hist(image)
                id = search_cosine(vector, 3)
                # id = search_euclidean(vector, 3)

                if id[0] != 'out':

                    print(f'Image: {filename} is similar to vidéo: {index[id[0]][0]} at {index[id[0]][1]}s that is {evaluate_result(row[1], index[id[0]][0])}')
                    result_csv.writerow([filename.replace('.jpeg', ''), index[id[0]][0], float(index[id[0]][1]) if row[2] else '0', evaluate_result(row[1], index[id[0]][0])])
                else:
                    print(f"Image: {filename} is not similar to any video {evaluate_result(row[1], 'out')}")
                    result_csv.writerow([filename.replace('.jpeg', ''), 'out', '', evaluate_result(row[1], 'out')])
        
        find_time = time.time() - start
    
    return index_time, find_time
