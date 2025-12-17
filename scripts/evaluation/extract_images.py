import numpy as np
import matplotlib.pyplot as plt
import os

# Création du dossier
save_dir = "test_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("Chargement des datasets...")
X_cancer = np.load("dataset_cancer.npy")
X_negative = np.load("dataset_negative.npy")

print(f"Extraction de 5 images de chaque classe dans '{save_dir}'...")

# Sauvegarder 5 Cancers
for i in range(5):
    img = X_cancer[i]
    # Matplotlib gère la sauvegarde en niveaux de gris
    plt.imsave(os.path.join(save_dir, f"CANCER_{i}.png"), img.squeeze(), cmap='gray')

# Sauvegarder 5 Sains
for i in range(5):
    img = X_negative[i]
    plt.imsave(os.path.join(save_dir, f"SAIN_{i}.png"), img.squeeze(), cmap='gray')

print("✅ Images extraites ! Tu peux maintenant les utiliser dans l'app.")