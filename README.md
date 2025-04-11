# Sistema de Reconocimiento de DNI Español

Este sistema permite extraer información de imágenes de DNI españoles mediante técnicas avanzadas de OCR (Reconocimiento Óptico de Caracteres) combinadas con análisis inteligente.

## Características

- Extracción precisa de nombres, apellidos, números de documento, fechas y otros datos relevantes
- Preprocesamiento de imágenes adaptado específicamente a DNI españoles
- Detección por regiones específicas para mayor precisión
- Análisis contextual para mejorar la precisión en campos críticos
- Combinación de OCR local (Tesseract) con API de OpenAI para resultados óptimos

## Requisitos

- Python 3.7 o superior
- Tesseract OCR 5.0 o superior (con soporte para idioma español)
- Bibliotecas Python: pytesseract, opencv-python, numpy, pillow, python-dateutil, python-dotenv

## Instalación

### 1. Instalación de Tesseract con soporte para español

#### Windows:
1. Descargar el instalador desde [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Durante la instalación, **seleccionar "Spanish" en la lista de idiomas adicionales**
3. Agregar Tesseract a la variable de entorno PATH (normalmente: `C:\Program Files\Tesseract-OCR`)

#### Linux (Debian/Ubuntu):
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-spa
```

#### macOS:
```bash
brew install tesseract
brew install tesseract-lang
```

### 2. Instalación de dependencias Python

```bash
pip install pytesseract pillow opencv-python python-dateutil python-dotenv
```

### 3. Configuración

Crear un archivo `.env` en el directorio raíz con las siguientes variables:
```
OPENAI_API_KEY=tu_clave_de_api_aquí
```

## Uso

```python
from IDAnalyzer import process_dni_image

# Procesar una imagen de DNI con OpenAI
resultado = process_dni_image("ruta/a/tu/imagen_dni.jpg", use_openai=True)
print(resultado)

# Procesar utilizando solo análisis local sin OpenAI
resultado_local = process_dni_image("ruta/a/tu/imagen_dni.jpg", use_openai=False)
print(resultado_local)
```

El resultado será un diccionario con los siguientes campos:
```python
{
    "Nombre": "NOMBRE_PERSONA",
    "Apellido": "APELLIDOS_PERSONA",
    "Documento": "12345678A",
    "FechaDeNacimiento": "01/01/1990",
    "TipoDocumento": "DNI",
    "Sexo": "M",
    "Nacionalidad": "ESP",
    "FechaValidez": "01/01/2030"
}
```

## Troubleshooting

### Si Tesseract no es encontrado en PATH:

Modificar la línea en `IDAnalyzer.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ajustar a tu ruta
```

### Problemas con el reconocimiento del español:

Verificar que los archivos de idioma español estén correctamente instalados:
- Windows: Carpeta `tessdata` dentro del directorio de instalación de Tesseract
- Linux: `/usr/share/tesseract-ocr/4.00/tessdata/`
- macOS: `/usr/local/share/tessdata/`

El archivo `spa.traineddata` debe estar presente.

## Optimización

Para mejorar el reconocimiento:
1. Utilizar imágenes de alta resolución (mínimo 300 DPI)
2. Asegurar iluminación uniforme sin sombras o reflejos
3. Capturar la imagen con el DNI completamente visible y sin inclinación

## Licencia

Este proyecto es de código abierto bajo la licencia MIT.
