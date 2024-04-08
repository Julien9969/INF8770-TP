from PIL import Image
import os

def calculate_average_jpeg_compression_ratio(folder_path):
    total_compression_ratio = 0
    total_images = 0

    # Parcourir tous les fichiers dans le répertoire spécifié
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Chemin complet de l'image
            image_path = os.path.join(folder_path, filename)

            # Ouvrir l'image avec Pillow
            with Image.open(image_path) as img:
                # Récupérer la taille originale de l'image en octets
                original_size = os.path.getsize(image_path)

                # Convertir l'image en JPEG avec une qualité maximale (100)
                img.convert("RGB").save("temp.jpg", format="JPEG", quality=100)

                # Récupérer la taille du fichier JPEG compressé
                compressed_size = os.path.getsize("temp.jpg")

                # Calculer le ratio de compression
                compression_ratio = original_size / compressed_size
                total_compression_ratio += compression_ratio
                total_images += 1

    # Calculer le taux de compression moyen
    if total_images > 0:
        average_compression_ratio = total_compression_ratio / total_images
        return average_compression_ratio
    else:
        return None

# Chemin vers le répertoire contenant les images JPEG
folder_path = "C:/Users/Alexandre/Desktop/INF8770-TP/TP3/data/jpeg"

# Calculer le taux de compression moyen des images JPEG dans le répertoire spécifié
average_ratio = calculate_average_jpeg_compression_ratio(folder_path)

if average_ratio is not None:
    print(f"Taux de compression moyen JPEG : {average_ratio:.2f}")
else:
    print("Aucune image JPEG trouvée dans le répertoire.")
