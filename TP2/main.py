from io import TextIOWrapper
import os
import numpy as np
import cv2
from KL_transform import kl_transform, quantification
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

LEVELS = [[8, 8, 8], [8, 8, 4], [8, 8, 0], [4, 4, 4]]

def calculate_compression_ratio(original_image_size, compressed_image_size):
    return original_image_size / compressed_image_size


def process_image(resultsFile: TextIOWrapper, RGB_YUV: str):
    i = 1
    for image_path in os.listdir('data'):
        if not image_path.endswith('.png'):
            continue

        image = cv2.imread(os.path.join('data', image_path))
        for level in LEVELS:
            print(f"\n\n{i}. Processing {image_path} with level: {str(level)}...", file=resultsFile)
            imageKL = kl_transform(os.path.join('data', image_path), level, show_img = False, color_space = RGB_YUV)

            params = [cv2.IMWRITE_PNG_COMPRESSION]

            cv2.imwrite(f'results/question2/{RGB_YUV}/{level[0]}{level[1]}{level[2]}_{image_path}', cv2.cvtColor(imageKL, cv2.COLOR_RGB2BGR if RGB_YUV == "RGB" else cv2.COLOR_YUV2BGR))
            cv2.imwrite(f'results/question2/temp.png', image)

            # original_size = os.path.getsize(os.path.join('data', image_path))
            original_size = os.path.getsize(f'results/question2/temp.png')
            compressed_size = os.path.getsize(f'results/question2/{RGB_YUV}/{level[0]}{level[1]}{level[2]}_{image_path}')
            compression_ratio = calculate_compression_ratio(original_size, compressed_size)

            psnr_val = psnr(image, imageKL)
            ssim_val = ssim(image, imageKL, multichannel=True, channel_axis=-1)

            print(f"   PSNR: {psnr_val:.3f} dB", file=resultsFile)
            print(f"   SSIM: {ssim_val:.4f}", file=resultsFile)
            print(f"   Compression Ratio: {1 - (level[0] + level[1] + level[2]) / 24:.4f}\n", file=resultsFile)
            i +=1

def question2():
    print('Question 2')

    try:
        os.makedirs('results/question2/RGB')
        os.makedirs('results/question2/YUV')
    except FileExistsError:
        pass

    with open('results/question2/RGB_results.txt', 'w') as resultsFile:
        process_image(resultsFile, "RGB")
    with open('results/question2/YUV_results.txt', 'w') as resultsFile:
        process_image(resultsFile, "YUV")



    
def question4():
    print('Question 4')

    try:
        os.makedirs('results/question3/img')
    except FileExistsError:
        pass

    with open('results/question3/q3_results.txt', 'w') as resultsFile:
        for image_path in os.listdir('data'):
            if not image_path.endswith('.png'):
                continue

            for image_path2 in os.listdir('data'):
                if not image_path2.endswith('.png'):
                    continue
                
                mixed_image = kl_transform(os.path.join('data', image_path), (8, 8, 4), show_img=False, color_space='RGB', isq4=True, imageQ4=os.path.join('data', image_path2))
                image2 = cv2.imread(os.path.join('data', image_path2)).astype(np.uint8)

                psnr_val = psnr(image2, mixed_image, data_range=255)
                ssim_val = ssim(image2, mixed_image, multichannel=True, channel_axis=-1, data_range=mixed_image.max() - mixed_image.min())

                cv2.imwrite(f'results/question3/img/{image_path[:-4]}_{image_path2[:-4]}.png', cv2.cvtColor(mixed_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'results/question3/temp.png', image2)
                sizef = os.path.getsize(f'results/question3/img/{image_path[:-4]}_{image_path2[:-4]}.png')
                sizeOrigibal = os.path.getsize(os.path.join('data', image_path))
                print(f"\n\nProcessing {image_path} and {image_path2}...", file=resultsFile)
                print(f"   PSNR: {psnr_val:.3f} dB", file=resultsFile)
                print(f"   SSIM: {ssim_val:.4f}", file=resultsFile)
                print(f"   Compression Ratio: { 1- (8 + 8 + 4) / 24:.4f}\n", file=resultsFile)


if __name__ == '__main__':
    question2()
    question4()
