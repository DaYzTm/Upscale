import numpy as np
from PIL import Image
import requests
from io import BytesIO
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
from realesrgan import RealESRGANer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use the pre-defined model for RealESRGAN_x4plus_anime_6B
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
model = model.to(device)

upsampler = RealESRGANer(
    scale=4,
    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
    model=model,
    tile=128,  # Установите размер тайла (например, 256)
    tile_pad=10,
    pre_pad=0,
    half=False,  # Установите half в False
    device=device
)

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def save_image(img, path):
    img.save(path)

def upscale_image(url, save_path):
    input_img = load_image_from_url(url)
    
    # Переведите изображение в numpy массив
    input_img_np = np.array(input_img)
    
    # Убедитесь, что изображение имеет диапазон [0, 255]
    input_img_np = input_img_np.astype(np.uint8)
    
    try:
        # Увеличьте изображение
        output_img, _ = upsampler.enhance(input_img_np)
        
        # Переведите результат обратно в изображение
        output_img = Image.fromarray(output_img)
        
        # Сохраните изображение
        save_image(output_img, save_path)
        print(f'Изображение сохранено в {save_path}')
    except RuntimeError as e:
        print(f'Ошибка: {e}')
    except UnboundLocalError as e:
        print(f'Ошибка: {e}')

# Пример использования
image_url = 'https://cdn.donmai.us/sample/e2/34/__original_drawn_by_ratatatat74__sample-e234a4fac1cf7d056d596ef64937cb8a.jpg'
save_path = 'output_image.png'

upscale_image(image_url, save_path)