import cv2
import tempfile
import os
import numpy as np
import pytesseract
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
import argparse
import shutil
from pathlib import Path
from IDAnalyzer import process_dni_image, convert_to_processable_image

# imports locales
from ImageAnalyzer import analyze_image
from IDAnalyzer import analyze_id

app = FastAPI()

def response_dni(file_path):
    """
    Procesa un DNI usando el método mejorado que soporta múltiples formatos
    """
    # Usar la función que soporta múltiples formatos
    id_json = process_dni_image(file_path, use_openai=True)
    print("Resultado del procesamiento:", id_json)
    return id_json

def response_ticket(image_path):
    text = analyze_image(image_path)
    print(text)
    id_json = analyze_ticket(text)
    print(id_json)
    return id_json

@app.post("/analyze-id")
async def data_from_id(file: UploadFile = File(...)):
    # Inicializar variables para archivos temporales
    temp_file_path = None
    processed_image_path = None
    
    try:
        # Crear directorio temporal si no existe
        temp_dir = os.path.join(tempfile.gettempdir(), "dni_ocr_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generar nombre único para archivo temporal con la extensión original
        file_extension = Path(file.filename).suffix.lower() if file.filename else ".unknown"
        temp_file_path = os.path.join(temp_dir, f"uploaded_file_{next(tempfile._get_candidate_names())}{file_extension}")
        
        # Guardar el archivo subido tal como viene
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"Archivo guardado temporalmente como: {temp_file_path}")
        
        # Usar el nuevo método que soporta múltiples formatos
        id_json = response_dni(temp_file_path)
        return JSONResponse(content=id_json, status_code=200)

    except Exception as e:
        print(f"Error detallado: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")
    finally:
        # Comentamos la limpieza de archivos temporales para depuración
        # try:
        #    if temp_file_path and os.path.exists(temp_file_path):
        #        os.remove(temp_file_path)
        #        print(f"Archivo temporal eliminado: {temp_file_path}")
        #    if processed_image_path and os.path.exists(processed_image_path):
        #        os.remove(processed_image_path)
        #        print(f"Imagen procesada eliminada: {processed_image_path}")
        # except Exception as e:
        #    print(f"Error al eliminar archivos temporales: {str(e)}")
        print(f"Archivos temporales conservados para depuración: {temp_file_path}")

@app.post("/analyze-ticket")
async def data_from_ticket(file: UploadFile = File(...)):
    # Inicializar variables para archivos temporales
    temp_file_path = None
    processed_path = None
    
    try:
        # Crear directorio temporal si no existe
        temp_dir = os.path.join(tempfile.gettempdir(), "dni_ocr_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generar nombre único para archivo temporal con la extensión original
        file_extension = Path(file.filename).suffix.lower() if file.filename else ".jpg"
        temp_file_path = os.path.join(temp_dir, f"ticket_{next(tempfile._get_candidate_names())}{file_extension}")
        
        # Guardar el archivo subido tal como viene
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Convertir a formato procesable si es necesario
        processed_path = convert_to_processable_image(temp_file_path)
        if processed_path:
            id_json = response_ticket(processed_path)
            # Comentamos la limpieza de la imagen procesada
            # try:
            #    if os.path.exists(processed_path):
            #        os.remove(processed_path)
            # except:
            #    pass
        else:
            # Si la conversión falla, intentar usar el archivo original
            id_json = response_ticket(temp_file_path)

        if id_json is None:
            raise HTTPException(status_code=422, detail="No se pudo procesar la imagen, introduzca una fotografía más clara")
        else:
            return JSONResponse(content=id_json, status_code=200)

    except Exception as e:
        print(f"Error detallado: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")
    finally:
        # Comentamos la limpieza de archivos temporales para depuración
        # if temp_file_path and os.path.exists(temp_file_path):
        #    try:
        #        os.remove(temp_file_path)
        #        print(f"Archivo temporal eliminado: {temp_file_path}")
        #    except Exception as e:
        #        print(f"Error al eliminar archivo temporal: {str(e)}")
        print(f"Archivos temporales conservados para depuración: {temp_file_path}, {processed_path}")

@app.get("/test")
def test(name = None):
    if name is None:
        return "Hola mundo"
    else:
        return "Hola" + " " + name

@app.get("/test-dni-moderno")
def test_dni_moderno():
    """
    Endpoint para probar el procesamiento con un DNI moderno simulado
    """
    try:
        # Crear directorio temporal si no existe
        temp_dir = os.path.join(tempfile.gettempdir(), "dni_ocr_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Ruta al archivo temporal
        temp_file_path = os.path.join(temp_dir, "dni_moderno_test.jpg")
        
        # Crear una imagen de prueba con los datos del DNI (simulando el DNI de la imagen)
        test_image = np.ones((400, 700, 3), dtype=np.uint8) * 255  # Fondo blanco
        
        # Simulamos texto del DNI (ESTEVE MORENO RAUL, DNI 07262594E, etc.)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_image, "APELLIDOS: ESTEVE MORENO", (50, 50), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "NOMBRE: RAUL", (50, 100), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "FECHA NACIMIENTO: 19 07 1996", (50, 150), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "DNI: 07262594E", (50, 200), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "SEXO: M", (50, 250), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "NACIONALIDAD: ESP", (50, 300), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "VALIDEZ: 21 09 2028", (50, 350), font, 0.7, (0, 0, 0), 2)
        
        # Guardar imagen de prueba
        cv2.imwrite(temp_file_path, test_image)
        
        # Procesar el DNI (sin usar OpenAI para la prueba)
        result = process_dni_image(temp_file_path, use_openai=False)
        
        # Resultado esperado para comparar
        expected = {
            "Nombre": "RAUL",
            "Apellido": "ESTEVE MORENO",
            "Documento": "07262594E",
            "FechaDeNacimiento": "19/07/1996",
            "TipoDocumento": "DNI",
            "Sexo": "M",
            "Nacionalidad": "ESP",
            "FechaValidez": "21/09/2028"
        }
        
        # Crear un diccionario con resultados comparados
        comparison = {}
        for key in expected:
            comparison[key] = {
                "esperado": expected.get(key, ""),
                "obtenido": result.get(key, ""),
                "coincide": expected.get(key, "") == result.get(key, "")
            }
        
        # Calcular porcentaje de precisión
        coincidencias = sum(1 for key in expected if expected.get(key, "") == result.get(key, ""))
        total_campos = len(expected)
        precision = (coincidencias / total_campos) * 100 if total_campos > 0 else 0
        
        # Comentamos la limpieza del archivo temporal
        # try:
        #    if os.path.exists(temp_file_path):
        #        os.remove(temp_file_path)
        # except:
        #    pass
        
        # Devolver resultado completo y la ruta del archivo temporal
        return {
            "resultado_procesamiento": result,
            "comparacion": comparison,
            "precision": f"{precision:.2f}%",
            "coincidencias": coincidencias,
            "total_campos": total_campos,
            "archivo_temporal": temp_file_path
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/test-dni-especifico")
def test_dni_especifico():
    """
    Endpoint para probar el procesamiento con un DNI específico simulado (Esteve Moreno Raul)
    Esta función está diseñada para probar las mejoras del sistema con la imagen específica
    que está dando problemas.
    """
    try:
        # Crear directorio temporal si no existe
        temp_dir = os.path.join(tempfile.gettempdir(), "dni_ocr_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Ruta al archivo temporal
        temp_file_path = os.path.join(temp_dir, "dni_especifico_test.jpg")
        
        # Crear una imagen de prueba con los datos del DNI específico
        test_image = np.ones((500, 800, 3), dtype=np.uint8) * 240  # Fondo gris claro (más similar al DNI real)
        
        # Simulamos texto del DNI (Raul Esteve Moreno)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Área de título
        cv2.putText(test_image, "REINO DE ESPANA", (250, 50), font, 0.9, (30, 30, 30), 2)
        cv2.putText(test_image, "DOCUMENTO NACIONAL DE IDENTIDAD", (150, 80), font, 0.6, (30, 30, 30), 1)
        
        # Datos del documento
        cv2.putText(test_image, "DNI 07262594E", (350, 120), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "APELLIDOS", (100, 160), font, 0.5, (100, 100, 100), 1)
        cv2.putText(test_image, "ESTEVE", (100, 190), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "MORENO", (100, 220), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "NOMBRE", (100, 250), font, 0.5, (100, 100, 100), 1)
        cv2.putText(test_image, "RAUL", (100, 280), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "SEXO", (100, 320), font, 0.5, (100, 100, 100), 1)
        cv2.putText(test_image, "M", (100, 350), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "NACIONALIDAD", (400, 320), font, 0.5, (100, 100, 100), 1)
        cv2.putText(test_image, "ESP", (400, 350), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "FECHA DE NACIMIENTO", (550, 380), font, 0.5, (100, 100, 100), 1)
        cv2.putText(test_image, "19 07 1996", (550, 410), font, 0.7, (0, 0, 0), 2)
        cv2.putText(test_image, "VALIDEZ", (550, 320), font, 0.5, (100, 100, 100), 1)
        cv2.putText(test_image, "21 09 2028", (550, 350), font, 0.7, (0, 0, 0), 2)
        
        # Guardar imagen de prueba
        cv2.imwrite(temp_file_path, test_image)
        
        # Realizar el procesamiento completo
        print(f"Procesando imagen de DNI específico: {temp_file_path}")
        
        # Primero procesamos normalmente
        result_normal = process_dni_image(temp_file_path, use_openai=False)
        
        # Guardar la ruta para depuración
        result_normal["_archivo_temp"] = temp_file_path
        
        # Resultado esperado para comparar
        expected = {
            "Nombre": "RAUL",
            "Apellido": "ESTEVE MORENO",
            "Documento": "07262594E",
            "FechaDeNacimiento": "19/07/1996",
            "TipoDocumento": "DNI",
            "Sexo": "M",
            "Nacionalidad": "ESP",
            "FechaValidez": "21/09/2028"
        }
        
        # Verificar los resultados
        coincidencias = sum(1 for key in expected if expected.get(key, "") == result_normal.get(key, ""))
        total_campos = len(expected)
        precision = (coincidencias / total_campos) * 100 if total_campos > 0 else 0
        
        # Crear un diccionario con resultados comparados
        comparison = {}
        for key in expected:
            comparison[key] = {
                "esperado": expected.get(key, ""),
                "obtenido": result_normal.get(key, ""),
                "coincide": expected.get(key, "") == result_normal.get(key, "")
            }
        
        # Devolver resultado completo y la ruta del archivo temporal
        return {
            "mensaje": "Prueba de procesamiento de DNI específico (Esteve Moreno Raul)",
            "resultado_normal": result_normal,
            "comparacion": comparison,
            "precision": f"{precision:.2f}%",
            "coincidencias": coincidencias,
            "total_campos": total_campos,
            "archivo_temporal": temp_file_path,
            "ruta_debug": os.path.join(tempfile.gettempdir(), "dni_ocr_regions_debug")
        }
    
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description='Procesar DNI español en varios formatos.')
    parser.add_argument('file_path', help='Ruta al archivo a procesar (imagen, PDF, DOCX)')
    parser.add_argument('--no-openai', action='store_true', help='No usar OpenAI (solo procesamiento local)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: El archivo {args.file_path} no existe")
        return
    
    use_openai = not args.no_openai
    
    print(f"Procesando DNI desde archivo: {args.file_path}")
    print(f"Usando OpenAI: {'No' if args.no_openai else 'Sí'}")
    
    # Procesar el documento
    result = process_dni_image(args.file_path, use_openai=use_openai)
    
    # Mostrar resultados
    print("\n===== RESULTADO =====")
    print(f"Nombre: {result.get('Nombre', '')}")
    print(f"Apellido: {result.get('Apellido', '')}")
    print(f"Documento: {result.get('Documento', '')}")
    print(f"Fecha de Nacimiento: {result.get('FechaDeNacimiento', '')}")
    print(f"Tipo de Documento: {result.get('TipoDocumento', '')}")
    print(f"Sexo: {result.get('Sexo', '')}")
    print(f"Nacionalidad: {result.get('Nacionalidad', '')}")
    print(f"Fecha de Validez: {result.get('FechaValidez', '')}")

def ejemplos():
    """
    Ejemplos de uso para diferentes formatos de archivo
    """
    print("Ejemplos de uso del sistema de reconocimiento de DNI:")
    
    # 1. Procesar una imagen directamente (JPG, PNG, TIFF, GIF)
    print("\n1. Procesar imagen:")
    print("python Main.py ruta_a_imagen.jpg")
    print("python Main.py ruta_a_imagen.png")
    print("python Main.py ruta_a_imagen.tiff")
    print("python Main.py ruta_a_imagen.gif")
    
    # 2. Procesar un PDF
    print("\n2. Procesar PDF:")
    print("python Main.py ruta_a_documento.pdf")
    
    # 3. Procesar un documento DOCX
    print("\n3. Procesar DOCX:")
    print("python Main.py ruta_a_documento.docx")
    
    # 4. Usar solo procesamiento local (sin OpenAI)
    print("\n4. Procesar sin usar OpenAI (solo procesamiento local):")
    print("python Main.py ruta_a_archivo.jpg --no-openai")
    
    # 5. Uso desde código Python
    print("\n5. Uso desde código Python:")
    print("""
from IDAnalyzer import process_dni_image

# Procesar una imagen con OpenAI
resultado = process_dni_image("ruta_a_imagen.jpg")
print(resultado)

# Procesar un PDF sin OpenAI (solo local)
resultado = process_dni_image("ruta_a_documento.pdf", use_openai=False)
print(resultado)
    """)

if __name__ == "__main__":
    # Si no se proporcionan argumentos, mostrar ejemplos
    import sys
    if len(sys.argv) == 1:
        ejemplos()
    else:
        main()