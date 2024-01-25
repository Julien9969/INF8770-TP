import matplotlib.pyplot as py
import numpy as np
from PIL import Image
from copy import deepcopy
import time


def LZW(Message: np.ndarray[int]):
    """
    Source: GITHUB des exemple du cours (https://github.com/gabilodeau/INF8770/blob/master/Codage%20LZW.ipynb)
    Fonction qui prend en entrée un message et qui le code avec l'algorithme LZW.
    Message: Numpy array de int
    """
    dictsymb =[[Message[0]]]
    dictbin = ["{:b}".format(0)]
    nbsymboles = 1
    for i in range(1, len(Message)):
        if Message[i] not in dictsymb:
            dictsymb += [[Message[i]]]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles +=1
            
    longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message)

    for i in range(nbsymboles):
        dictbin[i] = "{:b}".format(i).zfill(int(np.ceil(np.log2(nbsymboles))))
    
    dictsymb.sort()
    # dictionnaire = { key: bin for key, bin in list(zip(dictsymb, dictbin)) }
    dictionnaire = list(zip(dictsymb, dictbin))
    # print(dictionnaire) 

    i = 0
    MessageCode = []
    longueur = 0
    while i < len(Message):
        precsouschaine = [Message[i]] #sous-chaine qui sera codé
        souschaine = [Message[i]] #sous-chaine qui sera codé + 1 caractère (pour le dictionnaire)
        
        #Cherche la plus grande sous-chaine. On ajoute un caractère au fur et à mesure.
        while souschaine in dictsymb and i < len(Message):
            i += 1
            precsouschaine = deepcopy(souschaine)
            if i < len(Message):  #Si on a pas atteint la fin du message
                souschaine.append(Message[i])  

        #Codage de la plus grande sous-chaine à l'aide du dictionnaire 
        codebinaire = [dictbin[dictsymb.index(precsouschaine)]]
        MessageCode += codebinaire
        longueur += len(codebinaire[0]) 
        #Ajout de la sous-chaine codé + symbole suivant dans le dictionnaire.
        if i < len(Message):
            dictsymb += [souschaine]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles +=1
        
        #Ajout de 1 bit si requis
        if np.ceil(np.log2(nbsymboles)) > len(MessageCode[-1]):
            for j in range(nbsymboles):
                dictbin[j] = "{:b}".format(j).zfill(int(np.ceil(np.log2(nbsymboles))))
        

    # print(MessageCode)

    print("Longueur = {0}".format(longueur))
    print("Longueur originale = {0}".format(longueurOriginale))

    return longueur, longueurOriginale


def strMessageIntoInt(Message):
    Message = Message.encode('utf-8')
    Message = np.frombuffer(Message, np.uint8)
    return Message

if __name__ == "__main__":
    result = ""

    for i in range(1, 6):
        with open(f"data TP1/textes/texte_{i}.txt", "r", encoding='utf-8') as f:
            Message = f.read()
        
        Message = strMessageIntoInt(Message)
        start = time.time()
        longueur, longueurOriginale = LZW(Message)
        print(f"Temps d'execution : {time.time() - start: .3f} secondes")
        result += f"Texte {i}\n"
        result += f"Temps d'execution : {time.time() - start: .3f} secondes\n"
        result += f"Longueur = {longueur}, Longueur originale = {longueurOriginale}\n"
        result += f"Taux de compression = {1 - longueur/longueurOriginale}\n\n"
    

    result += "\n\n"
    for i in range(1, 6):
        img = Image.open(f'data TP1/images/image_{i}.png')
        m = np.frombuffer(img.tobytes(), np.uint8)
        start = time.time()
        longueur, longueurOriginale = LZW(m)
        print(f"Temps d'execution : {time.time() - start: .3f} secondes")
        result += f"Image {i}\n"
        result += f"Temps d'execution : {time.time() - start: .3f} secondes\n"
        result += f"Longueur = {longueur}, Longueur originale = {longueurOriginale}\n\n"
        result += f"Taux de compression = {1 - longueur/longueurOriginale}\n\n"
        
    with open("resultats-Alice codage.txt", "w") as f:
        f.write(result)
