import requests

def test_upscale(url):
    endpoint = "http://localhost:8000/upscale"
    payload = {"url": url}
    
    response = requests.post(endpoint, params=payload)
    
    if response.status_code == 200:
        with open("upscaled_image.png", "wb") as f:
            f.write(response.content)
        print("Image upscaled and saved as upscaled_image.png")
    else:
        print(f"Failed to upscale image. Status code: {response.status_code}, Detail: {response.json()}")

if __name__ == "__main__":
    # Замените URL на URL изображения, которое вы хотите протестировать
    test_image_url = "https://example.com/path/to/your/image.jpg"
    test_upscale(test_image_url)