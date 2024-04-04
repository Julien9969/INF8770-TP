import cv2, os, csv, time
import torch, pickle
from PIL import Image
from einops import rearrange
import torch
import torchvision.transforms as transforms
import torchvision.models as models

index:list[tuple[str, float]] = []
neural_net_matrix = torch.empty((0, 512), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) # 512 for resnet18
# neural_net_matrix = torch.empty((0, 2048), device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) # 2048 for resnet50
model = None
preprocess = transforms.Compose([
        transforms.Resize((224, 224)),                       # change la taille de l'image en 224x224
        transforms.ToTensor(),                              # convertit une image PIL ou numpy.ndarray (HxWxC) dans la plage [0, 255] en un torch.FloatTensor de forme (CxHxW) dans la plage [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # normalise les valeurs 
                            std=[0.229, 0.224, 0.225]),
    ])

IMG_FOLDER = 'data/jpeg'
VIDEO_FOLDER = 'data/mp4'
N_IMAGES_CLEFS = 24

F1_SCORE_NEU = 0.86

def index_build_neural_net(video_path = 'data/mp4/v001.mp4'):
    global index, neural_net_matrix

    vidcap = cv2.VideoCapture(video_path)
    # fps = vidcap.get(cv2.CAP_PROP_FPS)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 1
    
    while True:
        _, image = vidcap.read()

        if count % N_IMAGES_CLEFS != 0:
            count += 1
            continue
        try:
            img_desc = neu_desc(image)
        except:
            break 
        
        neural_net_matrix = torch.cat((neural_net_matrix, img_desc), 0)
        index.append([video_path[-8:-4], round(vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 6)])
        count += 1

    print(f'Indexed {video_path} with {num_frames} frames')
    torch.cuda.empty_cache() # libère la mémoire de la carte graphique



def load_model():
    global model
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)   # le modèle est chargé avec des poids pré-entrainés sur ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)   # le modèle est chargé avec des poids pré-entrainés sur ImageNet
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval();   

def neu_desc(input_batch):
    global model, preprocess
    input_batch = Image.fromarray(input_batch)
    input_tensor = preprocess(input_batch)  # 3 x 224 x 224
    input_tensor = input_tensor.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).unsqueeze(0)  # 1 x 3 x 224 x 224
    with torch.no_grad():
        output = model(input_tensor)  # 1 x 512 x 1 x 1 
        
    # torch.no_grad() permet de désactiver la conservation en mémoire des matrices d'activation nécessaires 
    # lors de la mise à jour des paramètres lors de l'apprentissage avec la rétropropagation des gradients. 
    # Cela permet de réduire la consommation de mémoire graphique.
    return torch.nn.functional.normalize(rearrange(output, 'b d h w -> b (d h w)'))  # 1 x 512

def neu_cosine(mat, image_to_find):
    return torch.matmul(mat, image_to_find.t())

def search_cosine_neu(image_to_find, top_k=1):
    global neural_net_matrix
    similarities = neu_cosine(neural_net_matrix, image_to_find)
    index_of_highest_value = torch.argmax(similarities)
    return [index_of_highest_value.item()] if similarities[index_of_highest_value] > F1_SCORE_NEU else ['out']

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

def neu_find(result_csv: csv.writer, folder=IMG_FOLDER):
    global index, neural_net_matrix
    load_model()
    index_time = None
    # save_vars_neu(101)
    print(f"F1_SCORE_NEU: {F1_SCORE_NEU}, N_IMAGES_CLEFS: {N_IMAGES_CLEFS}")
    try:
        raise FileNotFoundError
        load_vars_neu()
        index_time = 'loaded from file'
        pass
    except FileNotFoundError:
        start = time.time()
        save_vars_neu(101)
        index_time = time.time() - start

    print(f"nombre de frame dans l'index {len(index)}")

    with open('data/gt.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header

        start = time.time()
        for i, (row, filename) in enumerate(zip(reader, os.listdir(folder))):
            if filename.endswith(".jpeg"):
                image = cv2.imread(os.path.join(folder, filename))

                vector = neu_desc(image)
                id = search_cosine_neu(vector, 3)

                if id[0] != 'out':
                    print(f'Image: {filename} is similar to vidéo: {index[id[0]][0]} at {index[id[0]][1]}s that is {evaluate_result(row[1], index[id[0]][0])}')
                    result_csv.writerow([filename.replace('.jpeg', ''), index[id[0]][0], float(index[id[0]][1]) if row[2] else '0', evaluate_result(row[1], index[id[0]][0])])
                else:
                    print(f"Image: {filename} is not similar to any video {evaluate_result(row[1], 'out')}")
                    result_csv.writerow([filename.replace('.jpeg', ''), 'out', '', evaluate_result(row[1], 'out')])

        find_time = time.time() - start

    return index_time, find_time