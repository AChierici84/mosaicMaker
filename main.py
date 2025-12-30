import requests
import cv2
import numpy as np

from mosaicMaker.mosaic import mosaic_maker


def download_image(url):
    response = requests.get(url)
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# Esempio di utilizzo
if __name__ == "__main__":
    image_url = "https://www.focus.it/images/2021/11/23/mare_1020x680.jpg"  # Sostituisci con l'URL della tua immagine
    image_path = "mare_1020x680.jpg"
    
    # Scarica l'immagine
    image = download_image(image_url)
    cv2.imwrite(image_path, image)
    
    # Crea il mosaico
    mosaic_maker(image_path)
    

