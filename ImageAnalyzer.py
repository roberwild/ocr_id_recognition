import cv2
import pytesseract.pytesseract
import json
import re
from PIL import Image
from datetime import datetime
from dateutil import parser
import numpy as np
import pytesseract


#configuracion de la orientacion de la pagina y el modelo de ocr
custom_config = r"--psm 11 --oem 3"
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\rgutierrez\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# **
# Lee la imagen y la convierte a escala de grises y luego la hace binaria
# **
def preprocess_image(image_path):
    # Lee la imagen y la convierte a escala de grises
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #aumenta el contraste de la imagen
    # gray= cv2.equalizeHist(gray)


    # Aplica un filtro GaussianBlur ligero para reducir el ruido
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = np.ones((2, 2), np.uint8)
    # thresh_img = cv2.dilate(thresh_img, kernel, iterations=1)


    # Retorna la imagen procesada (sin invertir para mantener texto oscuro y fondo claro)
    return thresh_img


# **
# Extrae el texto de la imagen y crea una copia temporal de la imagen original
# **
def extract_text_from_image(image_path):
    #se crea una imagen temporal copia de la imagen original pero procesada anteriormente y se extrae el texto
   
    processed_img= preprocess_image(image_path)

    # data_image= pytesseract.image_to_data(processed_img, output_type=Output.DICT)
   
    # image_boxes(data_image, processed_img, image_path)
   
    temp_img_path = "imagen_temporal.jpg"
    cv2.imwrite(temp_img_path, processed_img)


    text = pytesseract.image_to_string(Image.open(temp_img_path), config=custom_config)

    return text

def extract_json(text):
    # Usar una expresión regular para encontrar el JSON en el texto
    match = re.search(r'{.*}', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        # Convertir el string JSON a un objeto Python (dict)
        return json.loads(json_str)
    else:
        return None  # Retornar None si no se encuentra un JSON

def analyze_date(date):
    try:
        # Analiza la fecha en varios formatos
        fecha = parser.parse(date)
    except (ValueError, TypeError):
        # Si ocurre un error, devolver 31 de diciembre de 1969
        print("Error al analizar la fecha, devolviendo fecha predeterminada.")
        fecha = datetime(1969, 12, 31)
    
    # Retorna la fecha en formato dd/mm/yyyy
    return fecha.strftime("%d/%m/%Y")

def analyze_image(image_path):
    text = extract_text_from_image(image_path)
    return text