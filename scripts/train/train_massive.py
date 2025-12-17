import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

print("--- GÉNÉRATION MASSIVE DE DONNÉES ---")

# 1. Chargement
X_cancer_raw = np.load("dataset_cancer.npy")
X_negative_raw = np.load("dataset_negative.npy")

print(f"Original -> Cancer: {len(X_cancer_raw)}, Négatif: {len(X_negative_raw)}")

# 2. Configuration de l'augmentation
datagen = ImageDataGenerator(
    rotation_range=45,      # Rotations plus fortes
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# 3. Fonction pour multiplier les images
def augment_data(images, multiplier):
    augmented_list = []
    for img in images:
        # On ajoute l'image originale
        augmented_list.append(img)
        # On génère des variations
        img = img.reshape((1,) + img.shape) # Format (1, 50, 50, 1)
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_list.append(batch[0])
            i += 1
            if i >= multiplier:
                break
    return np.array(augmented_list)

# On multiplie les cancers par 20 et les négatifs par 15 pour avoir environ 2500 images
print("Génération en cours (patience)...")
X_cancer_aug = augment_data(X_cancer_raw, 20)
X_negative_aug = augment_data(X_negative_raw, 15)

print(f"Nouveau Dataset -> Cancer: {len(X_cancer_aug)}, Négatif: {len(X_negative_aug)}")

# 4. Labels et Fusion
y_cancer = np.ones(len(X_cancer_aug))
y_negative = np.zeros(len(X_negative_aug))

X = np.concatenate((X_cancer_aug, X_negative_aug), axis=0)
y = np.concatenate((y_cancer, y_negative), axis=0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Entraînement sur {len(X_train)} images | Test sur {len(X_test)} images")

# 5. Modèle "Simple CNN" (Celui qui marchait bien)
model = Sequential()
model.add(Input(shape=(50, 50, 1)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Entraînement
print("\nDémarrage de l'entraînement MASSIF... 🚀")
history = model.fit(X_train, y_train, 
                    epochs=25, 
                    batch_size=32,
                    validation_data=(X_test, y_test))

# 7. Résultat
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Précision finale sur le Test Set : {acc*100:.2f}%")

# Graphiques
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Précision (Accuracy)')
plt.legend()
plt.show()

# Sauvegarde
model.save("modele_final_subset0.h5")