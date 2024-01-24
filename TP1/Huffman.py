import numpy as np
import math
from PIL import Image
import time
from anytree import Node, RenderTree, PreOrderIter, AsciiStyle
import base64

def Huffman(Message: str):
    """
    Source: GITHUB des exemple du cours (https://github.com/gabilodeau/INF8770/blob/master/Codage%20Huffman.ipynb)
    Fonction qui prend en entrée un message et qui le code avec l'algorithme de Huffman.
    Message: string
    """
    #Liste qui sera modifié jusqu'à ce qu'elle contienne seulement la racine de l'arbre
    ArbreSymb =[[Message[0], Message.count(Message[0]), Node(Message[0])]]
    #dictionnaire obtenu à partir de l'arbre.
    dictionnaire = [[Message[0], '']]
    nbsymboles = 1

    #Recherche des feuilles de l'arbre
    for i in range(1,len(Message)):
        if not list(filter(lambda x: x[0] == Message[i], ArbreSymb)):
            ArbreSymb += [[Message[i], Message.count(Message[i]),Node(Message[i])]]
            dictionnaire += [[Message[i], '']]
            nbsymboles += 1

    longueurOriginale = np.ceil(np.log2(nbsymboles))*len(Message)

    OccSymb = ArbreSymb.copy()

    #affichage des feuilles trouvées
    ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])
    #print("ARBRESYMB: ",ArbreSymb)

    while len(ArbreSymb) > 1:
        #Fusion des noeuds de poids plus faibles
        symbfusionnes = ArbreSymb[0][0] + ArbreSymb[1][0]
        #Création d'un nouveau noeud
        noeud = Node(symbfusionnes)
        temp = [symbfusionnes, ArbreSymb[0][1] + ArbreSymb[1][1], noeud]
        #Ajustement de l'arbre pour connecter le nouveau avec ses parents
        ArbreSymb[0][2].parent = noeud
        ArbreSymb[1][2].parent = noeud
        #Enlève les noeuds fusionnés de la liste de noeud à fusionner.
        del ArbreSymb[0:2]
        #Ajout du nouveau noeud à la liste et tri.
        ArbreSymb += [temp]
        #Pour affichage de l'arbre ou des sous-branches
        #print('\nArbre actuel:\n\n')
        for i in range(len(ArbreSymb)):
            if len(ArbreSymb[i][0]) > 1:
                #print(RenderTree(ArbreSymb[i][2], style=AsciiStyle()).by_attr())
                RenderTree(ArbreSymb[i][2], style=AsciiStyle()).by_attr()
        ArbreSymb = sorted(ArbreSymb, key=lambda x: x[1])
        #print(ArbreSymb)

    ArbreCodes = Node('')
    noeud = ArbreCodes
    #print([node.name for node in PreOrderIter(ArbreSymb[0][2])])
    parcoursprefix = [node for node in PreOrderIter(ArbreSymb[0][2])]
    parcoursprefix = parcoursprefix[1:len(parcoursprefix)] #ignore la racine

    Prevdepth = 0 #pour suivre les mouvements en profondeur dans l'arbre
    for node in parcoursprefix:  #Liste des noeuds
        if Prevdepth < node.depth: #On va plus profond dans l'arbre, on met un 0
            temp = Node(noeud.name + '0')
            noeud.children = [temp]
            if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        elif Prevdepth == node.depth: #Même profondeur, autre feuille, on met un 1
            temp = Node(noeud.name + '1')
            noeud.children = [noeud.children[0], temp]  #Ajoute le deuxième enfant
            if node.children: #On avance le "pointeur" noeud si le noeud ajouté a des enfants.
                noeud = temp
        else:
            for i in range(Prevdepth-node.depth): #On prend une autre branche, donc on met un 1
                noeud = noeud.parent #On remontre dans l'arbre pour prendre la prochaine branche non explorée.
            temp = Node(noeud.name + '1')
            noeud.children = [noeud.children[0], temp]
            if node.children:
                noeud = temp

        Prevdepth = node.depth

    #print('\nArbre des codes:\n\n',RenderTree(ArbreCodes, style=AsciiStyle()).by_attr())
    #print('\nArbre des symboles:\n\n', RenderTree(ArbreSymb[0][2], style=AsciiStyle()).by_attr())

    ArbreSymbList = [node for node in PreOrderIter(ArbreSymb[0][2])]
    ArbreCodeList = [node for node in PreOrderIter(ArbreCodes)]

    for i in range(len(ArbreSymbList)):
        if ArbreSymbList[i].is_leaf: #Génère des codes pour les feuilles seulement
            temp = list(filter(lambda x: x[0] == ArbreSymbList[i].name, dictionnaire))
            if temp:
                indice = dictionnaire.index(temp[0])
                dictionnaire[indice][1] = ArbreCodeList[i].name

    #print(dictionnaire)

    MessageCode = []
    longueur = 0
    for i in range(len(Message)):
        substitution = list(filter(lambda x: x[0] == Message[i], dictionnaire))
        MessageCode += [substitution[0][1]]
        longueur += len(substitution[0][1])

    #print(MessageCode)
    print("Longueur = {0}".format(longueur))
    print("Longueur originale = {0}".format(longueurOriginale))
    print('Espérance: ' + str(longueur/len(Message)))
    entropie =0
    for i in range(nbsymboles):
        entropie = entropie-(OccSymb[i][1]/len(Message))*math.log(OccSymb[i][1]/len(Message),2)

    print('Entropie: ' + str(entropie))
    print('Taux de compression: ' + str(1 - longueur/longueurOriginale))
    return longueur, longueurOriginale, entropie, 1 - longueur/longueurOriginale


def strMessageIntoInt(Message):
    Message = Message.encode('utf-8')
    Message = np.frombuffer(Message, np.uint8)
    return Message


if __name__ == "__main__":
    result =''
    for i in range(1, 6):
        with open(f'TP1/data TP1/textes/texte_{i}.txt', 'r', encoding='utf-8') as f:
            Message = f.read()
        print(f'\n -----CODAGE TU TEXTE {i}: ------')
        start = time.time()
        longueur, longueurOriginale, entropie, tauxCompression = Huffman(Message)
        result += f"Texte {i}\n"
        result += f"Longueur = {longueur}, Longueur originale = {longueurOriginale}\n"
        result += f"Entropie = {entropie}\n"
        result += f"Taux de compression = {tauxCompression}\n"
        result += f"Temps d'execution : {time.time() - start: .3f} secondes\n\n"



    for i in range(1, 6):
        # with open(f"TP1/data TP1/images/image_{i}.png", "rb") as image:
        #     m = base64.b64encode(image.read())
        img = Image.open(f'data TP1/images/image_{i}.png')
        message = ''
        m = np.frombuffer(img.tobytes(), np.uint8)
        for j in m:
            message += chr(j)
        #m = np.frombuffer(img.tobytes(), np.uint8)
        print(f"--- CODAGE IMAGE {i} ---")
        start = time.time()
        longueur, longueurOriginale, entropie, tauxCompression = Huffman(message)
        result += f"Image {i}\n"
        result += f"Longueur = {longueur}, Longueur originale = {longueurOriginale}\n"
        result += f"Entropie = {entropie}\n"
        result += f"Taux de compression = {tauxCompression}\n"
        result += f"Temps d'execution : {time.time() - start: .3f} secondes\n\n"

        # result += f"Image {i}\n"
        # result += f"Temps d'execution : {time.time() - start: .3f} secondes\n"
        # result += f"Longueur = {longueur}, Longueur originale = {longueurOriginale}\n\n"

    with open('TP1/resultatHuffman.txt', 'w') as f:
        f.write(result)