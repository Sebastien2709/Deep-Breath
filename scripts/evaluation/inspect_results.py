import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Chargement des données (Les mêmes que pour l'entraînement)
print("Chargement des données...")
X_cancer = np.load("dataset_cancer.npy") # Si tu as utilisé le dataset augmenté, charge-le ici
X_negative = np.load("dataset_negative.npy")

# (Optionnel) Si tu veux tester sur les données BRUTES (non augmentées) pour voir la "vraie" performance clinique
# C'est le test le plus honnête.
# Si tu n'as que les versions augmentées sous la main, utilise-les, mais c'est mieux avec les originales.

# Recréation des labels
y_cancer = np.ones(len(X_cancer))
y_negative = np.zeros(len(X_negative))

X = np.concatenate((X_cancer, X_negative), axis=0)
y = np.concatenate((y_cancer, y_negative), axis=0)

# On refait le split pour avoir le même Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Chargement du Champion
print("Chargement du modèle...")
model = tf.keras.models.load_model('meilleur_modele.keras')

# 3. Prédictions
print("Le modèle réfléchit... 🧠")
predictions = model.predict(X_test)
# Si > 0.5 alors Cancer (1), sinon Sain (0)
y_pred_classes = (predictions > 0.5).astype(int).flatten()

# 4. Affichage
plt.figure(figsize=(15, 10))
plt.suptitle(f"Résultats sur le Test Set (Vrai vs Prédiction)", fontsize=16)

# On affiche 16 images au hasard
indices = np.random.choice(len(X_test), 16, replace=False)

for i, idx in enumerate(indices):
    plt.subplot(4, 4, i + 1)
    
    img = X_test[idx]
    true_label = int(y_test[idx])
    pred_label = int(y_pred_classes[idx])
    confiance = predictions[idx][0]
    
    # Couleur du titre : Vert si correct, Rouge si erreur
    col = 'green' if true_label == pred_label else 'red'
    
    label_dict = {0: "Sain", 1: "CANCER"}
    
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Vrai: {label_dict[true_label]}\nPréd: {label_dict[pred_label]}\nConf: {confiance:.2f}", color=col)
    plt.axis('off')

plt.tight_layout()
plt.show()