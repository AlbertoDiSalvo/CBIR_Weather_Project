import streamlit as st
import numpy as np
import cv2
import faiss
import os
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from skimage.feature import local_binary_pattern


st.set_page_config(page_title="Sistema CBIR - Clima", layout="wide")
IMAGE_SIZE = (224, 224)
BASE_DIR = "weather"

# Cargar el modelo CNN una sola vez (Cache) para que sea rápido
@st.cache_resource
def load_cnn_model():
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return Model(inputs=base_model.input, outputs=base_model.output)

cnn_model = load_cnn_model()

# Funciones de extraccion, iguales a las de "codigo.ipynb"
def get_cnn_features(img_array):
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)
    features = cnn_model.predict(x, verbose=0).flatten()
    return features

def get_hsv_features(img_array):
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def get_lbp_features(img_array):
    RADIUS = 3
    N_POINTS = 8 * RADIUS
    METHOD = "uniform"
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    lbp_matrix = local_binary_pattern(gray, N_POINTS, RADIUS, method=METHOD)
    
    n_bins = int(lbp_matrix.max() + 1)
    hist, _ = np.histogram(lbp_matrix.ravel(), bins=n_bins, range=(0, n_bins))
    
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_query_features(img_pil, extractors_list):
    """
    Procesa la imagen subida y extrae el vector combinado
    """
    img_pil = img_pil.resize(IMAGE_SIZE)
    img_array = np.array(img_pil)
    features_list = []
    
    # Extraer
    if "CNN" in extractors_list:
        features_list.append(get_cnn_features(img_array))        
    if "HSV" in extractors_list:
        features_list.append(get_hsv_features(img_array))    
    if "LBP" in extractors_list:
        features_list.append(get_lbp_features(img_array))
        
    # Concatenar y Normalizar
    if features_list:
        final_vector = np.concatenate(features_list).astype("float32")
        faiss.normalize_L2(final_vector.reshape(1, -1))
        return final_vector.reshape(1, -1)
    return None

#Interfaz de usuario
st.title("🌦️ Buscador de Imágenes Climáticas (CBIR)")
st.markdown("Sube una imagen para encontrar fotos similares basadas en contenido visual.")

# Barra lateral
with st.sidebar:
    st.header("Configuración")
    
    # Selector de Modelo
    model_options = {
        "CNN + Color (HSV) + Textura (LBP)": "CNN_HSV_LBP",
        "CNN + Color (HSV)": "CNN_HSV",
        "CNN + Textura (LBP)": "CNN_LBP"
    }
    
    selected_option = st.selectbox("Descriptores a usar:", list(model_options.keys()))
    index_suffix = model_options[selected_option]
    
    # Definir extractores según la selección
    active_extractors = []
    if "CNN" in index_suffix: active_extractors.append("CNN")
    if "HSV" in index_suffix: active_extractors.append("HSV")
    if "LBP" in index_suffix: active_extractors.append("LBP")
    
    top_k = st.slider("Número de resultados:", 1, 10, 5)

# Carga de archivos
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen de consulta
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Tu consulta")
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, width="stretch")
        
    with col2:
        st.subheader("Resultados")
        
        # Cargar índice y rutas
        try:
            index = faiss.read_index(f"faiss_index_{index_suffix}.bin")
            paths = np.load(f"paths_{index_suffix}.npy")
        except Exception as e:
            st.error(f"Error cargando los archivos del índice {index_suffix}. Asegúrate de que los archivos .bin y .npy existen en la carpeta raíz.")
            st.stop()
            
        # Extraer características de query
        with st.spinner('Procesando imagen...'):
            query_vector = extract_query_features(image_pil, active_extractors)
        
        # Buscar en FAISS
        if query_vector is not None:
            distances, indices = index.search(query_vector, top_k)
            
            # Mostrar resultados
            cols = st.columns(top_k)
            
            for i, col in enumerate(cols):
                idx = indices[0][i]
                dist = distances[0][i]
                img_path = paths[idx]
                                
                with col:
                    if os.path.exists(img_path):
                        st.image(img_path, width="stretch")
                        class_name = img_path.split(os.sep)[-3] if os.sep in img_path else "Desconocido"
                        st.caption(f"**{class_name}**\nDist: {dist:.4f}")
                    else:
                        st.error(f"No encontrado:\n{img_path}")