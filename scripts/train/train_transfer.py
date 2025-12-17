import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- 1. Chargement et Préparation ---
print("Chargement des données...")
if not os.path.exists("dataset_cancer.npy"):
    print("ERREUR: Fichiers manquants.")
    exit()

X_cancer = np.load("dataset_cancer.npy")
X_negative = np.load("dataset_negative.npy")

# Labels
y_cancer = np.ones(X_cancer.shape[0])
y_negative = np.zeros(X_negative.shape[0])

# Fusion
X = np.concatenate((X_cancer, X_negative), axis=0)
y = np.concatenate((y_cancer, y_negative), axis=0)

# TRICHE INTELLIGENTE : VGG16 veut des images couleur (3 canaux : Rouge, Vert, Bleu).
# Nos scanners sont en noir et blanc (1 canal).
# On va copier notre image 3 fois pour faire croire à VGG que c'est de la couleur.
print("Conversion en format 'RGB' pour VGG16...")
X_rgb = np.repeat(X, 3, axis=-1) # Passe de (50,50,1) à (50,50,3)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_rgb, y, test_size=0.2, random_state=42)

# --- 2. Chargement du Cerveau Pré-entraîné (VGG16) ---
print("Téléchargement de VGG16 (ImageNet)...")
# include_top=False : On garde les yeux (convolutions) mais on jette le cerveau final (qui classe les chiens/chats)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(50, 50, 3))

# On "gèle" les premières couches pour ne pas casser ce que VGG sait déjà
for layer in base_model.layers:
    layer.trainable = False

# --- 3. Création du Modèle Hybride ---
model = Sequential()
model.add(base_model) # Les yeux experts
model.add(Flatten())  # On aplatit

# Notre cerveau spécialisé "Cancer"
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5)) # Pour éviter le par-cœur
model.add(Dense(1, activation='sigmoid')) # Verdict final

# On utilise un learning rate plus petit pour ne pas brusquer le modèle
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# --- 4. Entraînement ---
print("\nEntraînement du modèle Hybride... 🧠")
history = model.fit(X_train, y_train, 
                    epochs=30, 
                    batch_size=16, # Plus petit batch pour apprendre plus souvent
                    validation_data=(X_test, y_test))

# --- 5. Résultats ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🏆 Précision avec Transfer Learning : {acc*100:.2f}%")

# Courbes
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Performance VGG16')
plt.legend()
plt.show()

# Sauvegarde
model.save("modele_vgg16_cancer.h5")