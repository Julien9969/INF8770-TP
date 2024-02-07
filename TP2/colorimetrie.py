import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def rgb_to_yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def kl_transform(image):
    transformed_channels = [cv2.dct(image[:, :, i].astype(np.float32)) for i in range(image.shape[2])]
    return np.stack(transformed_channels, axis=2)

def inverse_kl_transform(transformed_image):
    inverse_transformed_channels = [cv2.idct(transformed_image[:, :, i]) for i in range(transformed_image.shape[2])]
    return np.stack(inverse_transformed_channels, axis=2)

def quantize_image(image, bits_per_channel):
    max_val = 2 ** bits_per_channel - 1
    return np.round(image / max_val) * max_val




image = cv2.imread('data/kodim01.png') 

image_yuv = rgb_to_yuv(image)

transformed_image = kl_transform(image_yuv)

quantized_images = []
for bits_per_channel in [(8, 8, 8), (8, 8, 4), (8, 8, 0)]:
    quantized_image = quantize_image(transformed_image, bits_per_channel)
    quantized_images.append(quantized_image)

for i, quantized_image in enumerate(quantized_images):
    reconstructed_image_yuv = inverse_kl_transform(quantized_image)
    reconstructed_image = cv2.cvtColor(reconstructed_image_yuv.astype(np.uint8), cv2.COLOR_YUV2RGB)

    psnr_val = psnr(image, reconstructed_image)
    ssim_val = ssim(image, reconstructed_image, multichannel=True)

    print(f"Configuration de bits {i+1}:")
    print(f"   PSNR: {psnr_val:.2f} dB")
    print(f"   SSIM: {ssim_val:.4f}")

    original_size = image.size
    compressed_size = quantized_image.size
    compression_ratio = original_size / compressed_size
    print(f"   Taux de compression: {compression_ratio:.2f}")

    cv2.imwrite(f"reconstructed_image_{i+1}.jpg", reconstructed_image)
