import cv2
import tempfile
import os
import numpy as np
import pytesseract.pytesseract
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# imports locales
from ImageAnalyzer import analyze_image
from IDAnalyzer import analyze_id



app = FastAPI()

def response_dni(image_path):
    text = analyze_image(image_path)
    print(text)
    id_json = analyze_id(text)
    print(id_json)
    return id_json

def response_ticket(image_path):
    text = analyze_image(image_path)
    print(text)
    id_json = analyze_ticket(text)
    print(id_json)
    return id_json


@app.post("/analyze-id")
async def data_from_id(file: UploadFile):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decodifica la imagen
        
        if img is None:
            raise HTTPException(status_code=422, detail="Error al decodificar la imagen")
        
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            cv2.imwrite(temp_file.name, img)
            temp_file_path = temp_file.name

        id_json = response_dni(temp_file_path)
        return JSONResponse(content=id_json, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")
    finally:
        if temp_file_path:
            os.remove(temp_file_path)


@app.post("/analyze-ticket")
async def data_from_ticket(file: UploadFile):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decodifica la imagen
        
        if img is None:
            raise HTTPException(status_code=422, detail="Error al decodificar la imagen")
        
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            cv2.imwrite(temp_file.name, img)
            temp_file_path = temp_file.name


        id_json = response_ticket(temp_file_path)

        if id_json is None:
            raise HTTPException(status_code=422, detail="No se pudo procesar la imagen, introduzca una fotografía más clara")
        else:
            return JSONResponse(content=id_json, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")
    finally:
        if temp_file_path:
            os.remove(temp_file_path)


@app.get("/test")
def test(name = None):
    if name is None:
        return "Hola mundo"
    else:
        return "Hola" + " " + name