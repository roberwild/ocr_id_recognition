import cv2
import numpy as np
import os
import tempfile
from IDAnalyzer import process_dni_image

# Creamos un directorio temporal si no existe
temp_dir = os.path.join(tempfile.gettempdir(), "dni_ocr_temp")
os.makedirs(temp_dir, exist_ok=True)

# Datos de la imagen hardcodeada (esto simula la imagen del DNI que el usuario envió)
# Estos valores se ajustarían manualmente según el DNI mostrado en la interfaz
temp_file_path = os.path.join(temp_dir, "dni_test.jpg")

# Crear una imagen de prueba con los datos del DNI
# En un entorno real, esta imagen sería la subida por el usuario
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
print(f"Imagen de prueba guardada en: {temp_file_path}")

# Procesar el DNI
print("Procesando imagen del DNI...")
result = process_dni_image(temp_file_path, use_openai=False)  # False para no usar OpenAI en la prueba

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

# Verificar si los resultados coinciden con lo esperado
coincidencias = 0
total_campos = len(expected)

for key, expected_value in expected.items():
    actual_value = result.get(key, "")
    if expected_value == actual_value:
        coincidencias += 1
        print(f"✅ {key}: Coincide")
    else:
        print(f"❌ {key}: No coincide (Esperado: {expected_value}, Obtenido: {actual_value})")

# Calcular porcentaje de precisión
precision = (coincidencias / total_campos) * 100
print(f"\nPrecisión del OCR: {precision:.2f}%")

# Comentamos la limpieza del archivo temporal para depuración
# try:
#     if os.path.exists(temp_file_path):
#         os.remove(temp_file_path)
#         print(f"Archivo temporal eliminado: {temp_file_path}")
# except Exception as e:
#     print(f"Error al eliminar archivo temporal: {str(e)}")

print(f"Archivo temporal conservado para depuración: {temp_file_path}") 