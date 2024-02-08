import os

import cv2
from KL_transform import kl_transform
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def question2():
    print('Question 2')

    try:
        os.makedirs('results/question2/RGB')
        os.makedirs('results/question2/YUV')
    except FileExistsError:
        pass

    for image_path in os.listdir('data'):
        if not image_path.endswith('.png'):
            continue

        image = cv2.imread(os.path.join('data', image_path))
        print(f"Processing {image_path}...")
        imageKL = kl_transform(os.path.join('data', image_path), show_img = False, color_space = 'RGB')

        psnr_val = psnr(image, imageKL)
        ssim_val = ssim(image, imageKL, multichannel=True, channel_axis=-1)

        print(f"   PSNR: {psnr_val:.2f} dB")
        print(f"   SSIM: {ssim_val:.4f}")

        original_size = image.size
        compressed_size = imageKL.size
        compression_ratio = original_size / compressed_size
        print(f"   Taux de compression: {compression_ratio:.2f}\n")

        cv2.imwrite(f'results/question2/RGB/{image_path}', cv2.cvtColor(imageKL, cv2.COLOR_RGB2BGR))
        # for i, img in enumerate(images):




















if __name__ == '__main__':
    question2()