import openai
import json
from PIL import Image
import re
from datetime import datetime, timedelta
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
            "\n6. In Spanish DNI cards, the ExpiryDate is often found after or near 'VALIDEZ' or after the document number/support number."
            "\n7. If multiple dates are present, the most future date is likely the ExpiryDate."
            "\n8. Pay special attention to finding the ExpiryDate even if it's not clearly labeled with 'VALIDEZ'."
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
        # Si ocurre un error, devolver cadena vacía en lugar de fecha por defecto
        print("Error al analizar la fecha, devolviendo cadena vacía.")
        return ""
    
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
    
    # Lista de palabras que NO deben estar en los apellidos
    palabras_excluidas = ["NUM", "SOPORT", "GOPORT", "DOCUMENTO", "NACIONAL", "IDENTIDAD", 
                         "ESPAÑA", "VALIDEZ", "DNI", "FECHA", "NACIMIENTO", "NOMBRE", "SEXO", 
                         "APELLIDOS", "APELLIDO"]
    
    # Buscar nombre después de "NOMBRE"
    name_match = re.search(r'NOMBRE\s*[:\s]*([A-ZÁÉÍÓÚÜÑa-záéíóúüñ]+)', text)
    if name_match:
        result["Name"] = name_match.group(1).strip()
    
    # Buscar apellidos después de "APELLIDOS" - recuperando ambos apellidos
    # Primero buscar líneas que contengan APELLIDOS
    apellidos_completos = None
    lineas = text.split('\n')
    
    # Buscar líneas específicas con APELLIDOS
    for i, linea in enumerate(lineas):
        # Si la línea contiene APELLIDOS
        if "APELLIDOS" in linea:
            # Extraer los apellidos de la misma línea
            apellidos_texto = linea.replace("APELLIDOS", "").strip()
            
            # Si no hay suficiente texto en la misma línea o parece incompleto, buscar en líneas siguientes
            if not apellidos_texto or len(apellidos_texto.split()) < 2:
                # Comprobar si hay más líneas disponibles
                if i+1 < len(lineas) and i+2 < len(lineas):
                    # Considerar que los apellidos pueden estar en 1 o 2 líneas siguientes
                    linea_siguiente = lineas[i+1].strip()
                    linea_siguiente2 = lineas[i+2].strip()
                    
                    # Si la línea siguiente parece un apellido (solo mayúsculas y sin palabras excluidas)
                    if re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', linea_siguiente) and not any(palabra in linea_siguiente for palabra in palabras_excluidas):
                        # Si la segunda línea también parece apellido, combinarlos
                        if re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', linea_siguiente2) and not any(palabra in linea_siguiente2 for palabra in palabras_excluidas):
                            apellidos_completos = f"{linea_siguiente} {linea_siguiente2}"
                        else:
                            apellidos_completos = linea_siguiente
                    elif apellidos_texto:
                        # Si ya hay texto en la línea de APELLIDOS, usarlo
                        apellidos_completos = apellidos_texto
            else:
                # Si hay suficiente texto en la misma línea, usarlo
                apellidos_completos = apellidos_texto
            
            # Si encontramos apellidos, salir del bucle
            if apellidos_completos:
                break
    
    # Si encontramos apellidos en el análisis por líneas, usarlos
    if apellidos_completos:
        result["Lastname"] = apellidos_completos
    else:
        # Si no, intentar con patrones regex más generales
        # Buscar apellidos completos (primer y segundo apellido)
        apellidos_match = re.search(r'APELLIDOS\s*[:\s]*([A-ZÁÉÍÓÚÑ\s]+)', text)
        if apellidos_match:
            apellidos_candidato = apellidos_match.group(1).strip()
            # Verificar que no contenga palabras excluidas
            if not any(palabra in apellidos_candidato for palabra in palabras_excluidas):
                result["Lastname"] = apellidos_candidato
        else:
            # Buscar patrones específicos de documentos españoles con dos apellidos
            for i, linea in enumerate(lineas):
                # Buscar líneas que parezcan contener exclusivamente apellidos (dos palabras en mayúsculas)
                if re.match(r'^[A-ZÁÉÍÓÚÑ]+\s+[A-ZÁÉÍÓÚÑ]+$', linea.strip()):
                    # Verificar que no sean palabras prohibidas
                    if not any(palabra in linea for palabra in palabras_excluidas + ["NOMBRE", "APELLIDOS", "NACIONALIDAD"]):
                        result["Lastname"] = linea.strip()
                        break
    
    # Buscar fecha de nacimiento (patrón común en DNI español: DD MM YYYY)
    dob_patterns = [
        r'FECHA DE NACIMIENTO\s*[:\s]*(\d{1,2})\s+(\d{1,2})\s+(\d{4})',  # Formato español
        r'FECHA[.\s]+NAC[.\s]+(?:.*?)(\d{1,2})[/\s.-](\d{1,2})[/\s.-](\d{4})',  # Variación
        r'NACIMIENTO[^0-9]*(\d{1,2})\s+(\d{1,2})\s+(\d{4})',  # Después de NACIMIENTO
        r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})',  # Números separados por espacios
        r'(\d{1,2})[/\s.-](\d{1,2})[/\s.-](\d{2,4})',  # Formatos generales
    ]
    
    fecha_nacimiento = None
    fecha_nacimiento_obj = None
    for pattern in dob_patterns:
        match = re.search(pattern, text)
        if match:
            day, month, year = match.groups()
            if len(year) == 2:
                year = '19' + year if int(year) > 50 else '20' + year
            # Verificamos que se trata de una fecha española (DD/MM/YYYY)
            if int(day) <= 31 and int(month) <= 12 and 1900 <= int(year) <= 2100:
                fecha_nacimiento = f"{day}/{month}/{year}"
                # Guardar también la fecha como objeto para comparaciones
                try:
                    fecha_nacimiento_obj = datetime(int(year), int(month), int(day))
                    result["DateOfBirth"] = fecha_nacimiento
                    break
                except (ValueError, TypeError):
                    # Si la fecha no es válida, continuamos con el siguiente patrón
                    continue
    
    # Búsqueda específica de fecha de validez después de la palabra VALIDEZ
    validez_fecha = None
    if "VALIDEZ" in text:
        # Obtener texto después de VALIDEZ (hasta 40 caracteres)
        validez_idx = text.find("VALIDEZ")
        if validez_idx >= 0:
            texto_despues = text[validez_idx:validez_idx+40]
            # Buscar patrón de fecha específicamente del formato DD MM YYYY
            fecha_match = re.search(r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', texto_despues)
            if fecha_match:
                day, month, year = fecha_match.groups()
                # Validar que sea una fecha razonable (no años extremadamente lejanos)
                if int(day) <= 31 and int(month) <= 12 and 2000 <= int(year) <= 2050:
                    validez_fecha = f"{day}/{month}/{year}"
                    result["ExpiryDate"] = validez_fecha
    
    # Si no encontramos fecha de validez específica, buscar cerca de texto indicativo
    if not result["ExpiryDate"]:
        # Buscar después de palabras clave como NUM SOPORT, BGM, etc.
        soport_matches = re.finditer(r'(NUM\s*[GSC]OPORT|BGM|[A-Z]{2,3}\d{5,})', text)
        for match in soport_matches:
            # Examinar el texto después del match hasta 40 caracteres
            texto_despues = text[match.end():min(match.end()+40, len(text))]
            
            # Buscar específicamente patrones como "26 03 2028" (con espacios) primero
            fecha_match = re.search(r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', texto_despues)
            if fecha_match:
                day, month, year = fecha_match.groups()
                # Validar la fecha: debe ser razonable y posterior a hoy
                if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and 2000 <= int(year) <= 2050:
                    try:
                        fecha_obj = datetime(int(year), int(month), int(day))
                        # Debe ser una fecha futura o cercana a hoy
                        if fecha_obj > datetime.now() - timedelta(days=365):
                            result["ExpiryDate"] = f"{day}/{month}/{year}"
                    except (ValueError, TypeError):
                        continue
    
    # Buscar patrones específicos para fecha de validez
    if not result["ExpiryDate"]:
        # Buscar únicamente fechas futuras en el texto con formato específico para años 20XX
        fechas_futuras = []
        patron_fecha_futura = r'(\d{1,2})\s+(\d{1,2})\s+20(\d{2})'
        for match in re.finditer(patron_fecha_futura, text):
            day, month, year_short = match.groups()
            year = "20" + year_short
            # Validar componentes de la fecha
            if 1 <= int(day) <= 31 and 1 <= int(month) <= 12 and 2000 <= int(year) <= 2050:
                try:
                    fecha_obj = datetime(int(year), int(month), int(day))
                    # Agregar si es fecha futura o cercana a hoy
                    if fecha_obj > datetime.now() - timedelta(days=365):
                        # Calcular posición en el texto
                        pos = match.start()
                        # Si está cerca de VALIDEZ o NUM SOPORT, darle mayor prioridad
                        prioridad = 0
                        if "VALIDEZ" in text[max(0, pos-20):pos]:
                            prioridad = 2  # Mayor prioridad si está después de VALIDEZ
                        elif any(palabra in text[max(0, pos-30):pos] for palabra in ["SOPORT", "BGM", "NUM"]):
                            prioridad = 1  # Prioridad media si está después de número de soporte
                        
                        fechas_futuras.append((fecha_obj, f"{day}/{month}/{year}", prioridad))
                except (ValueError, TypeError):
                    continue
        
        # Si encontramos fechas futuras, usar la que tenga mayor prioridad o sea más cercana
        if fechas_futuras:
            # Ordenar por prioridad (descendente) y luego por fecha (ascendente)
            fechas_futuras.sort(key=lambda x: (-x[2], x[0]))
            result["ExpiryDate"] = fechas_futuras[0][1]
    
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


def es_fecha_valida(dia, mes, anio):
    """Verifica si una fecha es válida y razonable para un documento de identidad"""
    try:
        # Validar componentes básicos
        if not (1 <= int(dia) <= 31 and 1 <= int(mes) <= 12):
            return False
            
        # Validar año razonable (no fechas extremas)
        anio_int = int(anio)
        if not (1900 <= anio_int <= 2100):
            return False
            
        # Verificar que la fecha sea válida (ej: no 31 de febrero)
        fecha_obj = datetime(anio_int, int(mes), int(dia))
        
        # Para fechas de validez, asegurarse que no sean extremadamente lejanas
        if anio_int > datetime.now().year + 30:
            return False
            
        return True
    except ValueError:
        return False


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

        if "Name" in extracted_data and extracted_data["Name"]:
            new_json["Nombre"] = extracted_data["Name"]
        else:
            new_json["Nombre"] = ""
            
        if "Lastname" in extracted_data and extracted_data["Lastname"]:
            new_json["Apellido"] = extracted_data["Lastname"]
        else:
            new_json["Apellido"] = ""
            
        if "DocumentNumber" in extracted_data and extracted_data["DocumentNumber"]:
            new_json["Documento"] = extracted_data["DocumentNumber"]
        else:
            new_json["Documento"] = ""
            
        if "DateOfBirth" in extracted_data and extracted_data["DateOfBirth"]:
            fechaDeNacimiento = analyze_date(extracted_data["DateOfBirth"])
            new_json["FechaDeNacimiento"] = fechaDeNacimiento
        else:
            new_json["FechaDeNacimiento"] = ""
            
        if "DocumentType" in extracted_data and extracted_data["DocumentType"]:
            tipoDocumento = analyze_id_type(extracted_data["DocumentType"])
            new_json["TipoDocumento"] = tipoDocumento
        else:
            new_json["TipoDocumento"] = ""
            
        if "Sex" in extracted_data and extracted_data["Sex"]:
            new_json["Sexo"] = extracted_data["Sex"]
        else:
            new_json["Sexo"] = ""
            
        if "Nationality" in extracted_data and extracted_data["Nationality"]:
            new_json["Nacionalidad"] = extracted_data["Nationality"]
        else:
            new_json["Nacionalidad"] = ""
            
        if "ExpiryDate" in extracted_data and extracted_data["ExpiryDate"]:
            fechaValidez = analyze_date(extracted_data["ExpiryDate"])
            new_json["FechaValidez"] = fechaValidez
        else:
            new_json["FechaValidez"] = ""

        return new_json
    else:
        return {
            "Nombre": "",
            "Apellido": "",
            "Documento": "",
            "FechaDeNacimiento": "",
            "TipoDocumento": "",
            "Sexo": "",
            "Nacionalidad": "",
            "FechaValidez": ""
        }

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
        
        # Búsqueda específica para DNI español si no se encontraron ciertos campos
        # Restaurar detección de sexo y nacionalidad
        if not openai_json.get("Sex") and "SEXO" in text:
            sexo_match = re.search(r'SEXO\s*[:\s]*([MF])', text)
            if sexo_match:
                openai_json["Sex"] = sexo_match.group(1)
            elif "M " in text:  # Identificar M como sexo masculino
                openai_json["Sex"] = "M"
        
        if not openai_json.get("Nationality") and "ESP" in text:
            openai_json["Nationality"] = "ESP"
        
        # Lista de palabras que NO deben estar en los apellidos
        palabras_excluidas_apellidos = ["NUM", "SOPORT", "GOPORT", "DOCUMENTO", "NACIONAL", "IDENTIDAD", 
                                       "ESPAÑA", "VALIDEZ", "DNI", "FECHA", "NACIMIENTO", "NOMBRE", "SEXO", 
                                       "APELLIDOS", "APELLIDO"]
        
        # Variables para almacenar fechas encontradas
        fechas_encontradas = []
        
        # Procesar todas las fechas en el texto para clasificarlas correctamente
        for match in re.finditer(r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', text):
            dia, mes, anio = match.groups()
            if es_fecha_valida(dia, mes, anio):
                try:
                    fecha_obj = datetime(int(anio), int(mes), int(dia))
                    # Obtener el contexto antes de la fecha
                    inicio_pos = max(0, match.start() - 30)
                    fin_pos = match.start()
                    contexto_antes = text[inicio_pos:fin_pos]
                    
                    # Determinar tipo de fecha según contexto
                    tipo_fecha = "desconocido"
                    if "NACIMIENTO" in contexto_antes:
                        tipo_fecha = "nacimiento"
                    elif "VALIDEZ" in contexto_antes:
                        tipo_fecha = "validez"
                    # Si es fecha futura, probablemente es de validez
                    elif fecha_obj > datetime.now():
                        tipo_fecha = "validez"
                    # Si está cerca de patrones típicos de fechas de validez
                    elif re.search(r'BGM\d+|SOPORT|NUM', contexto_antes):
                        tipo_fecha = "validez"
                    
                    fechas_encontradas.append({
                        "fecha": fecha_obj,
                        "texto": f"{dia}/{mes}/{anio}",
                        "tipo": tipo_fecha,
                        "posicion": match.start()
                    })
                except ValueError:
                    pass
        
        # DETECCIÓN DE APELLIDOS MEDIANTE BÚSQUEDA DE CONTEXTO
        # Para evitar el hardcoding, buscamos patrones comunes en los DNI
        
        # 1. Buscar después del texto "APELLIDOS"
        apellidos_completos = None
        lineas = text.split('\n')
        
        for i, linea in enumerate(lineas):
            if "APELLIDOS" in linea:
                # Extraer texto después de la palabra APELLIDOS
                apellido_texto = linea.replace("APELLIDOS", "").strip()
                
                # Si hay apellidos en la misma línea
                if apellido_texto and not any(p in apellido_texto for p in palabras_excluidas_apellidos):
                    # Verificar si es formato típico de apellido
                    if re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', apellido_texto):
                        apellidos_completos = apellido_texto
                
                # Si no hay texto en la línea o es insuficiente, buscar en líneas siguientes
                if not apellidos_completos and i + 1 < len(lineas):
                    linea_siguiente = lineas[i+1].strip()
                    # Si la siguiente línea parece un apellido
                    if re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', linea_siguiente) and not any(p in linea_siguiente for p in palabras_excluidas_apellidos):
                        apellidos_completos = linea_siguiente
                        
                        # Si hay más líneas, verificar si hay un segundo apellido
                        if i + 2 < len(lineas):
                            linea_siguiente2 = lineas[i+2].strip()
                            # Si la línea siguiente también parece un apellido
                            if re.match(r'^[A-ZÁÉÍÓÚÑ]+$', linea_siguiente2) and not any(p in linea_siguiente2 for p in palabras_excluidas_apellidos):
                                # Combinar ambos apellidos
                                apellidos_completos = f"{apellidos_completos} {linea_siguiente2}"
                break
        
        # 2. Si no encontramos con el método anterior, buscar líneas con formato de apellidos
        if not apellidos_completos:
            # Buscar líneas con texto en mayúsculas que no contengan palabras excluidas
            for i, linea in enumerate(lineas):
                # Si es una línea de solo mayúsculas y sin palabras excluidas
                if (re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', linea.strip()) and 
                    len(linea.strip()) > 3 and 
                    not any(p in linea.upper() for p in palabras_excluidas_apellidos)):
                    
                    # Verificar si la línea siguiente también parece un apellido
                    if i + 1 < len(lineas):
                        linea_siguiente = lineas[i+1].strip()
                        if (re.match(r'^[A-ZÁÉÍÓÚÑ\s]+$', linea_siguiente) and 
                            len(linea_siguiente) > 3 and 
                            not any(p in linea_siguiente.upper() for p in palabras_excluidas_apellidos)):
                            
                            # Si tenemos dos líneas seguidas que parecen apellidos, combinarlas
                            apellidos_completos = f"{linea.strip()} {linea_siguiente.strip()}"
                            break
                    
                    # Si solo encontramos una línea que parece apellido, usarla
                    apellidos_completos = linea.strip()
                    break
        
        # 3. Estrategia adicional: buscar patrones de dos palabras mayúsculas juntas
        if not apellidos_completos:
            # Buscar líneas con exactamente dos palabras en mayúsculas (formato típico: PRIMER SEGUNDO)
            for linea in lineas:
                palabras = linea.strip().split()
                if len(palabras) == 2:
                    if (all(re.match(r'^[A-ZÁÉÍÓÚÑ]+$', palabra) for palabra in palabras) and
                        not any(p in linea.upper() for p in palabras_excluidas_apellidos)):
                        apellidos_completos = linea.strip()
                        break
        
        # Actualizar el JSON con los apellidos encontrados
        if apellidos_completos:
            openai_json["Lastname"] = apellidos_completos
        
        # Clasificar y asignar fechas según su tipo
        fecha_nacimiento = None
        fecha_validez = None
        
        # Primero, buscar específicamente fechas etiquetadas
        for fecha in fechas_encontradas:
            if fecha["tipo"] == "nacimiento":
                fecha_nacimiento = fecha["texto"]
            elif fecha["tipo"] == "validez":
                fecha_validez = fecha["texto"]
        
        # Si no encontramos fecha de nacimiento etiquetada, buscar la más antigua
        if not fecha_nacimiento and fechas_encontradas:
            fechas_pasadas = [f for f in fechas_encontradas if f["fecha"] < datetime.now()]
            if fechas_pasadas:
                fechas_pasadas.sort(key=lambda x: x["fecha"])  # Ordenar de más antigua a más reciente
                fecha_nacimiento = fechas_pasadas[0]["texto"]
        
        # Si no encontramos fecha de validez etiquetada, buscar la más futura
        if not fecha_validez and fechas_encontradas:
            fechas_futuras = [f for f in fechas_encontradas if f["fecha"] > datetime.now()]
            if fechas_futuras:
                fechas_futuras.sort(key=lambda x: x["fecha"])  # Ordenar de más cercana a más lejana
                fecha_validez = fechas_futuras[0]["texto"]
            # Si no hay fechas futuras, buscar la más reciente que esté cerca de "VALIDEZ" o "BGM"
            else:
                for fecha in fechas_encontradas:
                    inicio_pos = max(0, fecha["posicion"] - 30)
                    contexto = text[inicio_pos:fecha["posicion"]]
                    if "VALIDEZ" in contexto or "BGM" in contexto or "SOPORT" in contexto:
                        fecha_validez = fecha["texto"]
                        break
        
        # Actualizar fechas en el JSON
        if fecha_nacimiento:
            openai_json["DateOfBirth"] = fecha_nacimiento
        
        if fecha_validez:
            openai_json["ExpiryDate"] = fecha_validez
        
        # Búsqueda específica para fecha de validez (formato DD MM YYYY)
        if not openai_json.get("ExpiryDate") and "VALIDEZ" in text:
            validez_idx = text.find("VALIDEZ")
            if validez_idx >= 0:
                texto_despues = text[validez_idx:validez_idx+40]
                # Buscar fecha en formato DD MM YYYY cerca de VALIDEZ
                validez_match = re.search(r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', texto_despues)
                if validez_match:
                    dia, mes, anio = validez_match.groups()
                    openai_json["ExpiryDate"] = f"{dia}/{mes}/{anio}"
        
        # IMPORTANTE: Asegurarse de que la fecha de validez no sea igual a la fecha de nacimiento
        if (openai_json.get("ExpiryDate") and openai_json.get("DateOfBirth") and 
            openai_json["ExpiryDate"] == openai_json["DateOfBirth"]):
            # Si son iguales, es probable un error - buscar otra fecha que pueda ser la de validez
            print("¡ALERTA! La fecha de validez es igual a la fecha de nacimiento. Buscando otra fecha...")
            # Buscar explícitamente cerca de VALIDEZ o después de SOPORT
            for patrón, contexto in [
                (r'VALIDEZ[^0-9]*(\d{1,2})\s+(\d{1,2})\s+(\d{4})', "VALIDEZ"),
                (r'BGM\d+\s+(\d{1,2})\s+(\d{1,2})\s+(\d{4})', "SOPORTE"),
                (r'SOPORT[^0-9]*(\d{1,2})\s+(\d{1,2})\s+(\d{4})', "SOPORTE")
            ]:
                match = re.search(patrón, text)
                if match:
                    dia, mes, anio = match.groups()
                    nueva_fecha = f"{dia}/{mes}/{anio}"
                    # Verificar que sea diferente a la fecha de nacimiento
                    if nueva_fecha != openai_json["DateOfBirth"]:
                        openai_json["ExpiryDate"] = nueva_fecha
                        print(f"Nueva fecha de validez encontrada cerca de {contexto}: {nueva_fecha}")
                        break
            
            # Si aún son iguales, dejar vacía la fecha de validez para evitar confusiones
            if openai_json.get("ExpiryDate") == openai_json.get("DateOfBirth"):
                openai_json["ExpiryDate"] = ""
                print("No se pudo encontrar una fecha de validez clara y distinta. Dejando campo vacío.")
        
        # Convertir y formatear el resultado
        id_json = final_json(openai_json)
        
        # VERIFICACIONES FINALES
        # Si falta nacionalidad para DNI español
        if not id_json.get("Nacionalidad") and id_json.get("TipoDocumento") == "DNI":
            id_json["Nacionalidad"] = "ESP"
        
        # Si falta sexo pero está en el texto
        if not id_json.get("Sexo") and "SEXO" in text:
            if "SEXO M" in text:
                id_json["Sexo"] = "M"
            elif "SEXO F" in text:
                id_json["Sexo"] = "F"
        
        # Verificación final para evitar que la palabra APELLIDOS aparezca como apellido
        if id_json.get("Apellido") == "APELLIDOS":
            id_json["Apellido"] = ""
        
        # Última comprobación para evitar que la fecha de validez sea igual a la fecha de nacimiento
        if id_json.get("FechaValidez") == id_json.get("FechaDeNacimiento"):
            # Buscar explícitamente la fecha después de VALIDEZ
            validez_match = re.search(r'VALIDEZ[^0-9]*(\d{1,2})\s+(\d{1,2})\s+(\d{4})', text)
            if validez_match:
                dia, mes, anio = validez_match.groups()
                id_json["FechaValidez"] = f"{dia}/{mes}/{anio}"
            else:
                # Si no se encuentra, dejar el campo vacío
                id_json["FechaValidez"] = ""
        
        print(id_json)
        return id_json
    except Exception as e:
        print(f"Error en análisis de ID: {str(e)}")
        # Fallback básico para evitar error completo
        return {
            "Nombre": "",
            "Apellido": "",
            "Documento": "",
            "FechaDeNacimiento": "",
            "TipoDocumento": "",
            "Sexo": "",
            "Nacionalidad": "",
            "FechaValidez": ""
        }