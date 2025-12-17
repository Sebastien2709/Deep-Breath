import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- 1. Chargement ---
print("Chargement des datasets...")
if not os.path.exists("dataset_cancer.npy") or not os.path.exists("dataset_negative.npy"):
    print("ERREUR: Fichiers manquants.")
    exit()

X_cancer = np.load("dataset_cancer.npy")
X_negative = np.load("dataset_negative.npy")

# --- 2. Labels ---
y_cancer = np.ones(X_cancer.shape[0])
y_negative = np.zeros(X_negative.shape[0])

# --- 3. Fusion ---
X = np.concatenate((X_cancer, X_negative), axis=0)
y = np.concatenate((y_cancer, y_negative), axis=0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# --- 4. DATA AUGMENTATION (LA MAGIE OPÈRE ICI) ---
# On va créer des variantes des images à chaque époque
datagen = ImageDataGenerator(
    rotation_range=20,      # Tourner l'image de 20 degrés max
    width_shift_range=0.1,  # Décaler horizontalement
    height_shift_range=0.1, # Décaler verticalement
    zoom_range=0.1,         # Zoomer un peu
    horizontal_flip=True,   # Miroir horizontal
    vertical_flip=True      # Miroir vertical
)

# Important : le générateur a besoin de connaitre les données
datagen.fit(X_train)

# --- 5. Modèle (Légèrement ajusté) ---
model = Sequential()
model.add(Input(shape=(50, 50, 1))) 

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) # Une couche en plus !
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- 6. Entraînement avec Augmentation ---
print("\nDémarrage de l'entraînement AVEC Data Augmentation... 🚀")

# On utilise flow() pour générer les images à la volée
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=50, # On augmente les époques car c'est plus dur d'apprendre
                    validation_data=(X_test, y_test))

# --- 7. Résultats ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Précision finale : {test_acc*100:.2f}%")

# Graphiques
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Précision')
plt.legend()
plt.show()