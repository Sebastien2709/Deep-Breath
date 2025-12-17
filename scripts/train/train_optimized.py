import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

print("--- OPTIMISATION DU MODÈLE (Subset0) ---")

# 1. Chargement & Génération Massive (On reprend ta méthode gagnante)
if not os.path.exists("dataset_cancer.npy"):
    print("ERREUR : Fichiers manquants.")
    exit()

X_cancer_raw = np.load("dataset_cancer.npy")
X_negative_raw = np.load("dataset_negative.npy")

# Configuration de l'augmentation
datagen = ImageDataGenerator(
    rotation_range=90,      # On ose plus de rotation (le cancer n'a pas de sens haut/bas)
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

def augment_data(images, multiplier):
    augmented_list = []
    for img in images:
        augmented_list.append(img)
        img = img.reshape((1,) + img.shape)
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_list.append(batch[0])
            i += 1
            if i >= multiplier:
                break
    return np.array(augmented_list)

print("Génération des données augmentées...")
# On pousse un peu plus l'augmentation pour compenser
X_cancer_aug = augment_data(X_cancer_raw, 25)      # x25
X_negative_aug = augment_data(X_negative_raw, 18)  # x18

# Labels et Fusion
y_cancer = np.ones(len(X_cancer_aug))
y_negative = np.zeros(len(X_negative_aug))
X = np.concatenate((X_cancer_aug, X_negative_aug), axis=0)
y = np.concatenate((y_cancer, y_negative), axis=0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dataset : {len(X_train)} train | {len(X_test)} test")

# 2. Le Modèle "Pro" (Avec BatchNormalization)
model = Sequential()
model.add(Input(shape=(50, 50, 1)))

# Bloc 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization()) # <-- Nouveauté : Stabilise les neurones
model.add(MaxPooling2D((2, 2)))

# Bloc 2
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization()) # <-- Nouveauté
model.add(MaxPooling2D((2, 2)))

# Bloc 3
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization()) # <-- Nouveauté
model.add(MaxPooling2D((2, 2)))

# Bloc 4 (On va plus profond pour chercher les 90%+)
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Classifieur
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # On garde le dropout pour éviter le par-coeur
model.add(Dense(1, activation='sigmoid'))

# Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Les Callbacks (L'intelligence de l'entraînement)
callbacks = [
    # Si la précision ne s'améliore pas pendant 5 époques, on divise le learning rate par 2
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
    
    # On sauvegarde SEULEMENT si c'est le meilleur score jamais atteint
    ModelCheckpoint('meilleur_modele.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    
    # Si ça ne s'améliore plus du tout après 12 époques, on arrête pour ne pas perdre de temps
    EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)
]

# 4. Entraînement
print("\nDémarrage de l'entraînement OPTIMISÉ... 🚀")
history = model.fit(X_train, y_train, 
                    epochs=50, # On peut mettre beaucoup, le EarlyStopping arrêtera si besoin
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks) # On branche les callbacks

# 5. Résultat
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Précision finale (Meilleur checkpoint) : {acc*100:.2f}%")

# Graphiques
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Précision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss')
plt.legend()
plt.show()