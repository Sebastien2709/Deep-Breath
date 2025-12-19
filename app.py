import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="AI Cancer Detector", page_icon="🫁", layout="centered")

# --- CHARGEMENT DU MODÈLE (Mis en cache pour être rapide) ---
@st.cache_resource
def load_model():

    return tf.keras.models.load_model('models/meilleur_modele.keras')

try:
    model = load_model()
except:
    st.error("Erreur : Le fichier 'meilleur_modele.keras' est introuvable.")
    st.stop()

# --- INTERFACE ---
st.title("🫁 Détection Précoce - Cancer Pulmonaire")
st.markdown("Ce prototype utilise un **CNN (Réseau de Neurones Convolutif)** entraîné sur le dataset **LUNA16** pour détecter les nodules suspects.")
st.write("---")

# Zone d'upload
uploaded_file = st.file_uploader("Glissez une image de scan (patch 50x50) ici...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Affichage de l'image
    image = Image.open(uploaded_file).convert('L') # Convertir en niveaux de gris
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Image analysée", width=150)

    # 2. Prétraitement (Comme lors de l'entraînement !)
    # On redimensionne en 50x50 au cas où
    img_resized = image.resize((50, 50))
    img_array = np.array(img_resized)
    
    # Normalisation (0 à 255 -> 0 à 1)
    img_array = img_array / 255.0
    
    # Reshape pour le modèle (1, 50, 50, 1)
    img_input = img_array.reshape(1, 50, 50, 1)

    # 3. Prédiction
    with col2:
        st.write("Analyse en cours...")
        progress_bar = st.progress(0)
        prediction = model.predict(img_input)[0][0] # Score entre 0 et 1
        progress_bar.progress(100)

        # 4. Résultat
        seuil = 0.50
        confiance = prediction * 100
        
        if prediction > seuil:
            st.error(f"⚠️ **DÉTECTION : NODULE SUSPECT**")
            st.metric(label="Probabilité de Cancer", value=f"{confiance:.2f}%", delta="Risque Élevé")
            st.write("Le modèle a repéré une structure irrégulière caractéristique.")
        else:
            st.success(f"✅ **DÉTECTION : TISSU SAIN**")
            st.metric(label="Probabilité de Cancer", value=f"{confiance:.2f}%", delta="- Risque Faible", delta_color="normal")
            st.write("Aucune anomalie détectée sur ce patch.")

    # Détails techniques (Expandable)
    with st.expander("Voir les détails techniques"):
        st.write(f"Score brut du modèle : {prediction:.8f}")
        st.write(f"Dimension image entrée : {img_input.shape}")

st.write("---")
st.caption("Projet Deep Learning - Démonstration à but éducatif uniquement.")