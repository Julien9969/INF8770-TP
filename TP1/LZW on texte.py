import matplotlib.pyplot as py
import numpy as np
from PIL import Image

def LZW(Message):
    dictsymb =[Message[0]]
    dictbin = ["{:b}".format(0)]
    nbsymboles = 1
    for i in range(1,len(Message)):
        if Message[i] not in dictsymb:
            dictsymb += [Message[i]]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles += 1
            
    longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message)    

    for i in range(nbsymboles):
        dictbin[i] = "{:b}".format(i).zfill(int(np.ceil(np.log2(nbsymboles))))
            
    dictsymb.sort()
    dictionnaire = np.transpose([dictsymb,dictbin])
    print(dictionnaire) 

    i=0;
    MessageCode = []
    longueur = 0
    while i < len(Message):
        precsouschaine = Message[i] #sous-chaine qui sera codé
        souschaine = Message[i] #sous-chaine qui sera codé + 1 caractère (pour le dictionnaire)
        
        #Cherche la plus grande sous-chaine. On ajoute un caractère au fur et à mesure.
        while souschaine in dictsymb and i < len(Message):
            i += 1
            precsouschaine = souschaine
            if i < len(Message):  #Si on a pas atteint la fin du message
                souschaine += Message[i]  

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
        

    print(MessageCode)

    print("Longueur = {0}".format(longueur))
    print("Longueur originale = {0}".format(longueurOriginale))


if __name__ == "__main__":
    Message = "ABAABAABACABBABCDAADACABABAAABAABBABABAABAAB"

    with open("data TP1/textes/texte_4.txt", "r", encoding='utf-8') as f:
        Message = f.read()

    LZW(Message)

    

    # byte_string = np.array(img).tobytes()

    # print(byte_string)
    # Message = byte_string.decode('latin-1')

    # img = Image.open('data TP1/images/image_1.png')

    # # imagelue = py.imread('data TP1/images/image_1.png')
    # # image = imagelue.astype('int32')

    # m = np.frombuffer(img.tobytes(), np.uint8)
    # s = []
    # for i in range(len(m)):
    #     s.append(m[i])


    # print(m[0:100])
    # LZW(m)
        # img = Image.open('data TP1/images/image_1.png')

# byte_string = np.array(img).tobytes()

# # Convert the byte string to a character string
# char_string = byte_string.decode('latin-1')