# Reconocimiento de Imágenes Climáticas Usando CBIR

Este proyecto implementa un sistema de Recuperación de Imágenes Basado en Contenido (CBIR) capaz de identificar patrones meteorológicos (lluvia, nieve, arcoíris, etc.) sin necesidad de metadatos, utilizando únicamente las características visuales de las imágenes.

## Características
- **Extracción de Características Híbrida:** Combina ResNet50 (Forma/Semántica) + HSV (Color) + LBP (Textura).
- **Búsqueda Rápida:** Utiliza indexación FAISS para recuperación eficiente.
- **Interfaz Gráfica:** Aplicación web interactiva construida con Streamlit.

## Instalación y Uso

1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/AlbertoDiSalvo/CBIR_Weather_Project.git]
   cd CBIR_Weather_System

2. Instalar dependencias:
   ```bash
    pip install -r requirements.txt

3. Ejecutar la interfaz de búsqueda:
   ```bash
    streamlit run app.py

4. (Opcional) Regenerar índices: Ejecutar el notebook codigo.ipynb para re-entrenar o generar nuevos índices.

## Estructura del Proyecto
app.py: Interfaz de usuario (Streamlit).

codigo.ipynb: Notebook con el pipeline de extracción, indexación y evaluación.

memoria.pdf: Memoria técnica detallada del proyecto.

faiss_index_*.bin: Índices vectoriales pre-calculados.
