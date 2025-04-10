import openai
import json
from PIL import Image
import re
from datetime import datetime
from typing import Annotated
from dateutil import parser
from ImageAnalyzer import extract_json, analyze_date
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_text_openai(text):
    try:
        prompt = (
            "Please extract the following information from this Spanish ID card (DNI) text data: "
            "Name, Lastname, DateOfBirth, DocumentType, DocumentNumber, Sex, Nationality, ExpiryDate. "
            "\n\nLook carefully for Spanish terms like 'NOMBRE' (Name), 'APELLIDOS' (Lastname), 'FECHA DE NACIMIENTO' (DateOfBirth), "
            "'DNI' followed by numbers (DocumentNumber), 'SEXO' (Sex), 'NACIONALIDAD' (Nationality), and 'VALIDEZ' (ExpiryDate). "
            "\n\nIMPORTANT RULES FOR DATES:"
            "\n1. For all dates, they will be in DD MM YYYY format with day first, month second, and year last."
            "\n2. The ExpiryDate (fecha de validez) is usually the most recent/future date on the document."
            "\n3. ExpiryDate is typically located near the word 'VALIDEZ' and is a future date (later than DateOfBirth)."
            "\n4. NEVER confuse ExpiryDate with DateOfBirth - they are different dates with different purposes."
            "\n5. In Spanish DNI cards, the ExpiryDate is always a future date, while DateOfBirth is always in the past."
            "\n\nMake sure to extract the DNI number correctly - it's usually at the bottom of the card and may end with a letter. "
            "The sex will be 'M' for male or 'F' for female. The nationality is usually a 3-letter code like 'ESP' for Spain. "
            "\n\nReturn ONLY a JSON object with the exact field names specified above, no additional explanations. "
            "Here is the extracted text from the ID card:\n\n"
            f"{text}"
        )

        # Llamada a la API de OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o", # o "gpt-4" para mayor precisión
            messages=[
                {"role": "system", "content": "You are an AI specialized in extracting structured data from Spanish ID documents (DNI). Always extract all visible fields and respond with valid complete JSON. Pay special attention to correctly identify the expiry date (fecha de validez) which is different from the date of birth."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        # Extraer la respuesta
        extracted_data = response.choices[0].message.content
        return extracted_data

    except Exception as e:
        print(f"Error al usar OpenAI: {str(e)}")
        return fallback_text_analysis(text)


def analyze_date(date):
    try:
        # Analiza la fecha en varios formatos
        fecha = parser.parse(date, dayfirst=True)  # Priorizar formato día/mes/año (europeo)
    except (ValueError, TypeError):
        # Si ocurre un error, devolver 31 de diciembre de 1969
        print("Error al analizar la fecha, devolviendo fecha predeterminada.")
        fecha = datetime(1969, 12, 31)
    
    # Retorna la fecha en formato dd/mm/yyyy
    return fecha.strftime("%d/%m/%Y")


def fallback_text_analysis(text):
    """Función alternativa para analizar texto cuando OpenAI falla"""
    # Buscamos patrones comunes en documentos de identidad
    result = {
        "Name": "",
        "Lastname": "",
        "DateOfBirth": "",
        "DocumentType": "DNI",
        "DocumentNumber": "",
        "Sex": "",
        "Nationality": "",
        "ExpiryDate": ""
    }
    
    # Buscar nombre después de "NOMBRE"
    name_match = re.search(r'NOMBRE\s*[:\s]*([A-ZÁÉÍÓÚÜÑa-záéíóúüñ]+)', text)
    if name_match:
        result["Name"] = name_match.group(1).strip()
    
    # Buscar apellidos después de "APELLIDOS" - expresión mejorada
    lastname_patterns = [
        r'APELLIDOS\s*[:\s]*([A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s]+)(?=\s*NOMBRE)',  # Busca hasta NOMBRE
        r'APELLIDOS\s*[:\s]*([A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s]+)',  # Cualquier texto después de APELLIDOS
    ]
    
    for pattern in lastname_patterns:
        lastname_match = re.search(pattern, text)
        if lastname_match:
            result["Lastname"] = lastname_match.group(1).strip()
            break
    
    # Buscar apellidos genéricos sin hardcoding
    if not result["Lastname"]:
        # Buscar cualquier texto que parezca apellido después de APELLIDOS
        apellido_match = re.search(r'APELLIDOS[:\s]+([A-ZÁÉÍÓÚÜÑa-záéíóúüñ\s]+)', text)
        if apellido_match:
            result["Lastname"] = apellido_match.group(1).strip()
    
    # Buscar fecha de nacimiento (patrón común en DNI español: DD MM YYYY)
    dob_patterns = [
        r'FECHA DE NACIMIENTO\s*[:\s]*(\d{1,2})\s+(\d{1,2})\s+(\d{4})',  # Formato español
        r'FECHA[.\s]+NAC[.\s]+(?:.*?)(\d{1,2})[/\s.-](\d{1,2})[/\s.-](\d{4})',  # Variación
        r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})',  # Números separados por espacios
        r'(\d{1,2})[/\s.-](\d{1,2})[/\s.-](\d{2,4})',  # Formatos generales
    ]
    
    for pattern in dob_patterns:
        match = re.search(pattern, text)
        if match:
            day, month, year = match.groups()
            if len(year) == 2:
                year = '19' + year if int(year) > 50 else '20' + year
            # Verificamos que se trata de una fecha española (DD/MM/YYYY)
            if int(day) <= 31 and int(month) <= 12:
                result["DateOfBirth"] = f"{day}/{month}/{year}"
            else:
                # Si parece invertida, la corregimos
                result["DateOfBirth"] = f"{month}/{day}/{year}"
            break
    
    # Buscar fecha de validez (patrón similar a la fecha de nacimiento)
    validez_patterns = [
        r'VALIDEZ\s*[:\s]*(\d{1,2})\s+(\d{1,2})\s+(\d{4})',  # Formato español
        r'VALID[EZ]*\s*[:\s]*(\d{1,2})\s+(\d{1,2})\s+(\d{4})',  # Variación
        r'VALIDEZ.*?(\d{1,2})[\s./]+(\d{1,2})[\s./]+(\d{4})',  # Más flexible, busca números cerca de VALIDEZ
    ]
    
    for pattern in validez_patterns:
        match = re.search(pattern, text)
        if match:
            day, month, year = match.groups()
            if len(year) == 2:
                year = '20' + year  # Asumimos que la fecha de validez es futura
            result["ExpiryDate"] = f"{day}/{month}/{year}"
            break
    
    # Si no encontramos fecha de validez, buscamos específicamente cerca de la palabra VALIDEZ
    if not result["ExpiryDate"] and "VALIDEZ" in text:
        # Intentar buscar números cerca de "VALIDEZ"
        text_after_validez = text[text.find("VALIDEZ"):]
        # Buscar patrones de números que puedan ser fechas
        fecha_match = re.search(r'(\d{1,2})[\s./]+(\d{1,2})[\s./]+(\d{4})', text_after_validez)
        if fecha_match:
            day, month, year = fecha_match.groups()
            result["ExpiryDate"] = f"{day}/{month}/{year}"
        else:
            # Buscar patrones de números que puedan ser fechas
            numeros = re.findall(r'\d+', text_after_validez)
            if len(numeros) >= 3:  # Si encontramos al menos 3 números, asumimos día, mes, año
                result["ExpiryDate"] = f"{numeros[0]}/{numeros[1]}/{numeros[2]}"
    
    # Buscar sexo (M o F)
    sexo_patterns = [
        r'SEXO\s*[:\s]*([MF])',
        r'SEX[O]*\s*[:\s]*([MF])',
    ]
    
    for pattern in sexo_patterns:
        match = re.search(pattern, text)
        if match:
            result["Sex"] = match.group(1).strip()
            break
    
    # Buscar nacionalidad (generalmente ESP para DNI español)
    nacionalidad_patterns = [
        r'NACIONALIDAD\s*[:\s]*([A-Z]{3})',
        r'NATION[ALIDAD]*\s*[:\s]*([A-Z]{3})',
    ]
    
    for pattern in nacionalidad_patterns:
        match = re.search(pattern, text)
        if match:
            result["Nationality"] = match.group(1).strip()
            break
    
    # Buscar número de DNI (formato español: 8 dígitos + letra)
    dni_patterns = [
        r'DNI\s*[:\s]*([0-9]{7,8}[A-Za-z]?)',  # DNI + números + posible letra
        r'[0-9]{7,8}[A-Za-z]',  # Formato típico de DNI español
        r'[0-9]{6,}',  # Secuencia de al menos 6 dígitos como fallback
    ]
    
    for pattern in dni_patterns:
        match = re.search(pattern, text)
        if match:
            if match.groups():
                result["DocumentNumber"] = match.group(1).strip()
            else:
                result["DocumentNumber"] = match.group(0).strip()
            break
    
    # Si encontramos información, devolvemos un JSON formateado
    return json.dumps(result, ensure_ascii=False)


def analyze_id_type(id_type):
    id_type = id_type.lower()  # Convertimos todo el texto a minúsculas para facilitar comparaciones
    
    if "documento nacional de identidad" in id_type or "dni" in id_type or "documento" in id_type or re.search(r'documento', id_type):
        return "DNI"
    
    elif "cni" in id_type or "carte nationale d'identité" in id_type or "carte nationale" in id_type or re.search(r'cédula', id_type):
        return "CNI"
    
    elif "ci" in id_type or "cédula de identidad" in id_type or re.search(r'cedula', id_type):
        return "CI"
    
    elif "cpf" in id_type or "cadastro de pessoas fisicas" in id_type or "cadastro de pessoas físicas" in id_type:
        return "CPF"
    
    elif "pasaporte" in id_type or "passport" in id_type:
        return "PASAPORTE"
    
    else:
        return "OTRO"
    

def final_json(extracted_data):
    if(extracted_data):
        new_json = {}

        if "Name" in extracted_data:
            new_json["Nombre"] = extracted_data["Name"]
        if "Lastname" in extracted_data:
            new_json["Apellido"] = extracted_data["Lastname"]
        if "DocumentNumber" in extracted_data:
            new_json["Documento"] = extracted_data["DocumentNumber"]
        if "DateOfBirth" in extracted_data:
            fechaDeNacimiento = analyze_date(extracted_data["DateOfBirth"])
            new_json["FechaDeNacimiento"] = fechaDeNacimiento
        if "DocumentType" in extracted_data:
            tipoDocumento = analyze_id_type(extracted_data["DocumentType"])
            new_json["TipoDocumento"] = tipoDocumento
        if "Sex" in extracted_data:
            new_json["Sexo"] = extracted_data["Sex"]
        if "Nationality" in extracted_data:
            new_json["Nacionalidad"] = extracted_data["Nationality"]
        if "ExpiryDate" in extracted_data:
            fechaValidez = analyze_date(extracted_data["ExpiryDate"])
            new_json["FechaValidez"] = fechaValidez

        return new_json
    else:
        return "No se encontró un JSON válido."

def analyze_id(text):
    try:
        openai_text = analyze_text_openai(text)
        print(openai_text)
        openai_json = extract_json(openai_text)
        print(openai_json)
        
        # Si no pudimos extraer un JSON, intentamos con una versión alternativa
        if not openai_json:
            print("No se pudo extraer JSON del texto. Usando método alternativo.")
            # Intentar extraer información básica directamente del texto
            openai_json = fallback_text_analysis(text)
            if isinstance(openai_json, str):
                try:
                    openai_json = json.loads(openai_json)
                except:
                    openai_json = {
                        "Name": "",
                        "Lastname": "",
                        "DateOfBirth": "",
                        "DocumentType": "DNI",
                        "DocumentNumber": "",
                        "Sex": "",
                        "Nationality": "",
                        "ExpiryDate": ""
                    }
        
        # Correcciones para mejorar fechas sin hardcoding
        
        # Buscar fechas de nacimiento y validez en el texto
        fechas_encontradas = re.findall(r'(\d{1,2})[\s./]+(\d{1,2})[\s./]+(\d{4})', text)
        fechas_ordenadas = []
        
        for dia, mes, anio in fechas_encontradas:
            try:
                # Convertir a objeto datetime para comparar
                fecha_actual = datetime(int(anio), int(mes), int(dia))
                fechas_ordenadas.append(fecha_actual)
            except (ValueError, TypeError):
                continue
        
        # Ordenamos las fechas de más antigua a más reciente
        fechas_ordenadas.sort()
        
        # Búsqueda específica para DNI español si no se encontraron ciertos campos
        if not openai_json.get("Sex") and "SEXO" in text:
            sexo_match = re.search(r'SEXO\s*[:\s]*([MF])', text)
            if sexo_match:
                openai_json["Sex"] = sexo_match.group(1)
            elif "M " in text:  # Identificar M como sexo masculino
                openai_json["Sex"] = "M"
        
        if not openai_json.get("Nationality") and "ESP" in text:
            openai_json["Nationality"] = "ESP"
        
        id_json = final_json(openai_json)
        
        # Verificación final para fecha de validez
        if "FechaValidez" in id_json and "FechaDeNacimiento" in id_json:
            if id_json["FechaValidez"] == id_json["FechaDeNacimiento"]:
                # Si la fecha de validez es igual a la fecha de nacimiento, es un error
                # Buscar la fecha más reciente (probablemente es la de validez)
                if fechas_ordenadas and len(fechas_ordenadas) > 1:
                    fecha_mas_reciente = fechas_ordenadas[-1]  # La última es la más reciente
                    id_json["FechaValidez"] = fecha_mas_reciente.strftime("%d/%m/%Y")
        
        # Si no encontramos fecha de validez pero tenemos fechas ordenadas
        if ("FechaValidez" not in id_json or not id_json["FechaValidez"]) and fechas_ordenadas:
            # La fecha más reciente suele ser la fecha de validez
            if len(fechas_ordenadas) > 1:
                fecha_mas_reciente = fechas_ordenadas[-1]
                fecha_actual = datetime.now()
                
                # Si es una fecha futura, probablemente es la fecha de validez
                if fecha_mas_reciente > fecha_actual:
                    id_json["FechaValidez"] = fecha_mas_reciente.strftime("%d/%m/%Y")
                # Si no es futura, buscamos específicamente cerca de "VALIDEZ"
                elif "VALIDEZ" in text:
                    text_after_validez = text[text.find("VALIDEZ"):]
                    for dia, mes, anio in re.findall(r'(\d{1,2})[\s./]+(\d{1,2})[\s./]+(\d{4})', text_after_validez):
                        id_json["FechaValidez"] = f"{dia}/{mes}/{anio}"
                        break
        
        print(id_json)
        return id_json
    except Exception as e:
        print(f"Error en análisis de ID: {str(e)}")
        # Fallback básico para evitar error completo
        return {
            "Mensaje": "Error en el procesamiento",
            "Error": str(e),
            "TextoExtraido": text[:100] + "..." if len(text) > 100 else text
        }