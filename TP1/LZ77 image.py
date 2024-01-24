import matplotlib.pyplot as py
import numpy as np
from PIL import Image
from copy import deepcopy
import time
import Huffman


def rfind_sublist(lst, sublst):
    if len(sublst) > len(lst):
        return -1
    for i in range(len(lst) - len(sublst), -1, -1):
        if np.array_equal(lst[i:i + len(sublst)], sublst):
            return i
    return -1

def LZ77(message:np.ndarray[int], tailleDict:int):
    # tailleDict = 6  # Taille du dictionnaire (0,1,2,3,...)
    triplets = []  # Pour sauvegarder les triplets

    index = 0  # Position courante dans le message
    while index < len(message):
        decalage = 0
        symbDict = message[max(0, index - tailleDict): index]  # Symboles disponibles du dictionnaire
        if len(symbDict) < tailleDict:
            decalage = tailleDict - len(symbDict) #Pour l'ajustement des indices de position quand le dictionnaire n'est pas plein.
        # print(symbDict)
        # sousChaine = message[index:len(message)-1] #Le dernier caractère sera ajouté comme 3e élément du dernier triplet
        sousChaine = message[index:min(index+tailleDict, len(message) -1)] #Le dernier caractère sera ajouté comme 3e élément du dernier triplet

        # On cherche la sous-chaine la plus longue
        pos, length = 0, 0
        while len(sousChaine) > 0:
            # if sousChaine in symbDict:
            if symbDict.size > 0 and rfind_sublist(symbDict, sousChaine) != -1:
            # if symbDict.size > 0 and np.any(sousChaine == symbDict):
                # pos, length = symbDict.rfind(sousChaine), len(sousChaine) #dernière occurrence si plusieurs choix
                pos, length = rfind_sublist(symbDict, sousChaine), len(sousChaine) #dernière occurrence si plusieurs choix
                break
            sousChaine = sousChaine[:-1] # On a pas trouvé, donc on enlève un élément.
        
        # Regarde ensuite si on peut allonger la sous-chaine trouvée après la position de l'index: AB|ABABABABAB
        increment = 0
        while length > 0 \
                and index+length+increment < (len(message)-1) \
                and message[index-len(symbDict)+pos+length+increment] == message[index+length+increment]:
            increment += 1
        length += increment

        # Enregistrement des triplets
        c = message[index + length]  #Caractère suivant non encodé.

        if length == 0:
            pos = 0
        else:
            pos = pos + decalage #Pour l'ajustement des indices de position quand le dictionnaire n'est pas plein.

        triplets.append((pos, length, c))

        index += max(length+1, 1)  # Avance la position dans le message

        # print(triplets)
    return len(triplets) * 3, len(message), triplets


def strMessageIntoInt(Message):
    Message = Message.encode('utf-8')
    Message = np.frombuffer(Message, np.uint8)
    return Message

if __name__ == '__main__':
    result = ""

    for i in range(1, 6):
        with open(f"data TP1/textes/texte_{i}.txt", "r", encoding='utf-8') as f:
            Message = f.read()
        
        Message = strMessageIntoInt(Message)
        dictSize = 6

        start = time.time()
        longueur, longueurOriginale, triplets = LZ77(Message, dictSize)
        print(f"Temps d'execution : {time.time() - start: .3f} secondes")
        
        result += f"--- Texte {i} ---\n"
        result += f"Temps d'execution : {time.time() - start: .3f} secondes\n"
        result += f"Longueur = {longueur}, Longueur originale = {longueurOriginale}\n"
        result += f'Taux de compression: {str(1 - longueur/longueurOriginale)}\n\n'

    result += "\n\n"
    for i in range(1, 6):
        print(f"--- Image {i} ---")
        img = Image.open(f'data TP1/images/image_{i}.png')
        m = np.frombuffer(img.tobytes(), np.uint8)
        start = time.time()
        longueur, longueurOriginale, triplets = LZ77(m, 6)

        message = ""
        for triplet in triplets:
            message += chr(triplet[0])
            message += chr(triplet[1])
            message += chr(triplet[2])

        print(f"Temps d'execution : {time.time() - start: .3f} secondes")
        result += f"--- LZ77 Image {i} ---\n"
        result += f"Temps d'execution : {time.time() - start: .3f} secondes\n"
        result += f"Longueur = {longueur}, Longueur originale = {longueurOriginale}\n"
        result += f'Taux de compression: {str(1 - longueur/longueurOriginale)}\n\n'

        print("LZ77 done")
        Huffman.Huffman(message)
        result += f"\n--- LZ77 + Huffman Image {i} ---\n"
        result += f"Longueur = {longueur}, Longueur originale = {longueurOriginale}\n"
        result += f'Taux de compression: {str(1 - longueur/longueurOriginale)}\n\n'
        

    with open("resultats-LZ77.txt", "w") as f:
        f.write(result)

