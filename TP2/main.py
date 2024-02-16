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
            print(f"   Compression Ratio: {compression_ratio:.4f}\n", file=resultsFile)
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



    
def question3():
    print('Question 3')

    try:
        os.makedirs('results/question3/img')
    except FileExistsError:
        pass

    with open('results/question3/q3_results.txt', 'w') as resultsFile:
        for image_path in os.listdir('data'):
            if not image_path.endswith('.png'):
                continue
            
            imageKl = kl_transform(os.path.join('data', image_path), (8, 8, 8), show_img=False, color_space='RGB')

            for image_path2 in os.listdir('data'):
                if not image_path2.endswith('.png'):
                    continue
                
                image2 = cv2.cvtColor(cv2.imread(os.path.join('data', image_path2)), cv2.COLOR_BGR2RGB).astype('double').reshape(len(imageKl) * len(imageKl[0]), 3)
                image2 = quantification(image2, (8, 8, 8))
                image2 = image2.reshape(len(imageKl), len(imageKl[0]), 3)

                def mix_images(image1, image2):
                    image2 = image2.astype(image1.dtype)
                    avg_color = np.mean(image1, axis=(0, 1))

                    # Create a mask with the average color
                    mask = np.zeros_like(image1)
                    mask[:] = avg_color
                    mask = mask.astype(image1.dtype)

                    # Blend the images using weighted sum
                    mixed_image = cv2.addWeighted(image2, 0.5, mask, 0.5, 0)

                    return mixed_image
                
                mixed_image = mix_images(imageKl, image2)

                # Convert mixed_image to uint8 for PSNR calculation
                mixed_image_uint8 = np.clip(mixed_image, 0, 255).astype(np.uint8)

                # Calculate PSNR
                psnr_val = psnr(image2.astype(np.uint8), mixed_image_uint8, data_range=255)

                # Calculate SSIM
                ssim_val = ssim(image2, mixed_image, multichannel=True, channel_axis=-1, data_range=mixed_image.max() - mixed_image.min())

                # Write results to the file
                cv2.imwrite(f'results/question3/img/{image_path[:-4]}_{image_path2[:-4]}.png', cv2.cvtColor(mixed_image, cv2.COLOR_RGB2BGR))
                sizef = os.path.getsize(f'results/question3/img/{image_path[:-4]}_{image_path2[:-4]}.png')
                sizeOrigibal = os.path.getsize(os.path.join('data', image_path))
                print(f"\n\nProcessing {image_path} and {image_path2}...", file=resultsFile)
                print(f"   PSNR: {psnr_val:.3f} dB", file=resultsFile)
                print(f"   SSIM: {ssim_val:.4f}", file=resultsFile)
                print(f"   Compression Ratio: {sizeOrigibal/sizef:.4f}\n", file=resultsFile)


if __name__ == '__main__':
    # question2()
    question3()