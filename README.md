# OCR-ID Recognition

Sistema de reconocimiento óptico de caracteres (OCR) para extraer información de documentos de identidad, especialmente DNI español.

## Características

- Extracción de datos de DNI español (nombre, apellidos, número de documento, fecha de nacimiento, etc.)
- Procesamiento de imágenes para mejorar el reconocimiento
- Análisis de fechas en diferentes formatos
- Uso de API de OpenAI para análisis avanzado de texto
- Salida en formato JSON estructurado

## Requisitos

- Python 3.6+
- OpenCV
- OpenAI API
- Pillow
- Pytesseract
- Python-dateutil

## Instalación

```bash
git clone https://github.com/roberwild/ocr_id_recognition.git
cd ocr_id_recognition
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar el archivo .env con tu clave API de OpenAI
```

## Uso

```python
from IDAnalyzer import analyze_id

# Procesar texto extraído de una imagen de DNI
resultado = analyze_id(texto_extraido)
print(resultado)
```

## Estructura del proyecto

- `IDAnalyzer.py`: Módulo principal para análisis de documentos de identidad
- `ImageAnalyzer.py`: Funciones para procesamiento de imágenes
- `Main.py`: Archivo principal para ejecutar ejemplos
- `.env.example`: Plantilla para configurar variables de entorno

## Limitaciones actuales

- Optimizado principalmente para DNI español
- Requiere buena calidad de imagen para resultados óptimos
