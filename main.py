import requests
import numpy as np
from PIL import Image
import torch
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import psutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы
    allow_headers=["*"],  # Разрешаем все заголовки
)

# Установите устройство (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Оптимизация использования процессора
torch.set_num_threads(4)  # Установите количество потоков на значение по умолчанию

# Пути к моделям
model_paths = {
    "anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "real": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth"
}

# Функция для создания модели
def create_upsampler(model_path, model_type):
    if model_type == "anime":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    model = model.to(device)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=256,  # Установите размер тайла
        tile_pad=10,
        pre_pad=0,
        half=False,  # Убедитесь, что half-precision не используется
        device=device
    )
    return upsampler

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def save_image(img, path):
    img.save(path)

def upscale_image(url, model_type):
    input_img = load_image_from_url(url)
    
    # Переведите изображение в numpy массив
    input_img_np = np.array(input_img)
    
    # Убедитесь, что изображение имеет диапазон [0, 255]
    input_img_np = input_img_np.astype(np.uint8)
    
    upsampler = create_upsampler(model_paths[model_type], model_type)
    
    try:
        # Увеличьте изображение
        output_img, _ = upsampler.enhance(input_img_np)
        
        # Переведите результат обратно в изображение
        output_img = Image.fromarray(output_img)
        
        return output_img
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f'Runtime Error: {e}')
    except UnboundLocalError as e:
        raise HTTPException(status_code=500, detail=f'UnboundLocalError: {e}')

@app.post("/upscale")
def upscale_image_endpoint(url: str, model_type: str = "real"):
    if model_type not in model_paths:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'anime' or 'real'.")
    
    try:
        output_img = upscale_image(url, model_type)
        buf = BytesIO()
        output_img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stress")
def stress_endpoint():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    return {
        "cpu_usage": cpu_usage,
        "memory_total": memory_info.total,
        "memory_available": memory_info.available,
        "memory_used": memory_info.used,
        "memory_percentage": memory_info.percent
    }

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9500)