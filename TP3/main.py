# from datasets import load_dataset
from base64 import b64decode
from cv2.typing import MatLike
import cv2, os, csv
import numpy as np
import matplotlib.pyplot as plt
# import torchvision
import torch, pickle, time
import scipy.spatial.distance as dist
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
from einops import rearrange
from IPython.display import display
import matplotlib.pyplot as plt

BIN = 12

# F1 score https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
index:list[tuple[str, float]] = []
hist_matrix: list[MatLike] = []
neural_net_matrix = torch.empty((0, 512), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
model = None
preprocess = transforms.Compose([
        transforms.Resize((224, 224)),                       # change la taille de l'image en 224x224
        transforms.ToTensor(),                              # convertit une image PIL ou numpy.ndarray (HxWxC) dans la plage [0, 255] en un torch.FloatTensor de forme (CxHxW) dans la plage [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # normalise les valeurs 
                            std=[0.229, 0.224, 0.225]),
    ])

IMG_FOLDER = 'data/jpeg'
VIDEO_FOLDER = 'data/mp4'

F1_SCORE_HIST = 0.91
F1_SCORE_NEU = 0.86

def index_build_hist(video_path = 'data/mp4/v001.mp4'):
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

def index_build_neural_net(video_path = 'data/mp4/v001.mp4'):
    global index, neural_net_matrix

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    # plt.show() 
    count = 1
    
    while True:
        _, image = vidcap.read()
        
        try:
            img_desc = neu_desc(image)
        except:
            break 
        
        # neural_net_matrix.append(img_desc)
        neural_net_matrix = torch.cat((neural_net_matrix, img_desc), 0)
        index.append([video_path[-8:-4], round(vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 6)])
        count += 1

    print(f'Indexed {video_path} with {num_frames} frames')

def image_hist(image):
    red = cv2.calcHist([image], [2], None, [BIN], [0, 255])
    green = cv2.calcHist([image], [1], None, [BIN], [0, 255])
    blue = cv2.calcHist([image], [0], None, [BIN], [0, 255])
    hist = np.concatenate((red, green, blue), axis=0)

    # hist = cv2.calcHist([image], [0,1,2], None, [BIN, BIN, BIN], [0, 255, 0, 255, 0, 255])
    cv2.normalize(hist, hist)
    return hist.flatten()

def load_model():
    global model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # le modèle est chargé avec des poids pré-entrainés sur ImageNet
    # model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval();   

def neu_desc(input_batch):
    global model, preprocess
    input_batch = Image.fromarray(input_batch)
    input_tensor = preprocess(input_batch)  # 3 x 224 x 224
    input_tensor = input_tensor.unsqueeze(0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))  # 1 x 3 x 224 x 224
    with torch.no_grad():
        output = model(input_tensor)  # 1 x 512 x 1 x 1 
        
    # torch.no_grad() permet de désactiver la conservation en mémoire des matrices d'activation nécessaires 
    # lors de la mise à jour des paramètres lors de l'apprentissage avec la rétropropagation des gradients. 
    # Cela permet de réduire la consommation de mémoire graphique.
    return torch.nn.functional.normalize(rearrange(output, 'b d h w -> b (d h w)'))  # 1 x 512

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def neu_cosine(mat, image_to_find):
    return torch.matmul(mat, image_to_find.t())
     
def search_cosine(image_to_find, isNeural=False, top_k=1):
    if isNeural:
        global neural_net_matrix
        matrix = neural_net_matrix
    else:
        global hist_matrix
        matrix = hist_matrix
    
    distances = []
    for i, vector in enumerate(matrix):
        # print(vector.size(), image_to_find.size())
        if isNeural:
            similarity = neu_cosine(image_to_find, vector.unsqueeze(0))
        else:
            similarity = cosine(image_to_find, np.array(vector))
        distances.append((i, similarity))  # Store index and similarity

    distances.sort(key=lambda x: x[1], reverse=True)
    print(distances[:1])

    top_indices = [index if distance > F1_SCORE_HIST else 'out' for index, distance in distances[:top_k]]
    return top_indices

def search_cosine_neu(image_to_find, top_k=1):
    global neural_net_matrix
    similarities = neu_cosine(neural_net_matrix, image_to_find)
    index_of_highest_value = torch.argmax(similarities)
    return [index_of_highest_value.item()] if similarities[index_of_highest_value] > F1_SCORE_NEU else ['out']
    # return top_indices if top_indices else ['out']

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

def load_vars_neu():
    global index, neural_net_matrix
    with open('data/index_neu.pkl', 'rb') as load_index:
        index = pickle.load(load_index)

    with open('data/neu_matrix.pkl', 'rb') as load_hist:
        neural_net_matrix = pickle.load(load_hist)

def save_vars_neu(range_end=100):
    global index, neural_net_matrix

    for i in range(1, range_end):
        index_build_neural_net(f'{VIDEO_FOLDER}/v{i:03d}.mp4')

    with open('data/index_neu.pkl', 'wb') as save_index:
        pickle.dump(index, save_index)

    with open('data/neu_matrix.pkl', 'wb') as save_hist:
        pickle.dump(neural_net_matrix, save_hist)

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
    save_vars(101)
    # load_vars()
    # print(index[0:5])
    # print(hist_matrix[0:5])

    print(f"nombre de frame dans l'index {len(index)}")

    with open('data/gt.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header

        for i, (row, filename) in enumerate(zip(reader, os.listdir(folder))):
            # print(row,i, filename)
            # if i > 10: 
            #     break
            if filename.endswith(".jpeg"):
                image = cv2.imread(os.path.join(folder, filename))

                vector = image_hist(image)
                id = search_cosine(vector, True, 3)
                # id = search_euclidean(vector, 3)
                # print(id)

                if id[0] != 'out':

                    print(f'Image: {filename} is similar to vidéo: {index[id[0]][0]} at {index[id[0]][1]}s that is {evaluate_result(row[1], index[id[0]][0])}')
                    # result_csv.writerow([filename, index[id[0]][0], time_delta(float(index[id[0]][1]), float(row[2])) if row[2] else '0', evaluate_result(row[1], index[id[0]][0])])
                    result_csv.writerow([filename.replace('.jpeg', ''), index[id[0]][0], float(index[id[0]][1]) if row[2] else '0', evaluate_result(row[1], index[id[0]][0])])
                else:
                    print(f"Image: {filename} is not similar to any video {evaluate_result(row[1], 'out')}")
                    result_csv.writerow([filename.replace('.jpeg', ''), 'out', '', evaluate_result(row[1], 'out')])


def QUESTION2(result_csv: csv.writer, folder=IMG_FOLDER):
    global index, neural_net_matrix
    load_model()

    # save_vars_neu(101)
    load_vars_neu()
    # print(index[0:5])
    # print(hist_matrix[0:5])

    print(f"nombre de frame dans l'index {len(index)}")

    with open('data/gt.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header

        for i, (row, filename) in enumerate(zip(reader, os.listdir(folder))):
            # print(row,i, filename)
            # if i > 10: 
            #     break
            if filename.endswith(".jpeg"):
                image = cv2.imread(os.path.join(folder, filename))

                vector = neu_desc(image)
                # plt.plot(vector.cpu().numpy().flatten())
                # plt.title(f"Descripteur d'une image à l'aide de ResNet-18")
                # plt.show()
                id = search_cosine_neu(vector, 3)

                if id[0] != 'out':
                    print(f'Image: {filename} is similar to vidéo: {index[id[0]][0]} at {index[id[0]][1]}s that is {evaluate_result(row[1], index[id[0]][0])}')
                    # result_csv.writerow([filename, index[id[0]][0], time_delta(float(index[id[0]][1]), float(row[2])) if row[2] else '0', evaluate_result(row[1], index[id[0]][0])])
                    result_csv.writerow([filename.replace('.jpeg', ''), index[id[0]][0], float(index[id[0]][1]) if row[2] else '0', evaluate_result(row[1], index[id[0]][0])])
                else:
                    print(f"Image: {filename} is not similar to any video {evaluate_result(row[1], 'out')}")
                    result_csv.writerow([filename.replace('.jpeg', ''), 'out', '', evaluate_result(row[1], 'out')])


if __name__ == '__main__':
    print(BIN)
    # start = time.time()
    # with open('result.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['image', 'video_pred', 'minutage_pred', "evaluation"]) # Evaluation is TP, FP, TN, FN
    #     QUESTION1(writer)
    # print(f"Execution time: {time.time() - start} seconds")


    start = time.time()
    with open('result_neu.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image', 'video_pred', 'minutage_pred', "evaluation"]) # Evaluation is TP, FP, TN, FN
        QUESTION2(writer)
    print(f"Neural Execution time: {time.time() - start} seconds")

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