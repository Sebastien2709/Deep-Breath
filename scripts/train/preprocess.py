import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from tqdm import tqdm # Barre de progression

# --- CONFIG ---
base_path = "."  # Dossier courant
subset_path = os.path.join(base_path, "subset0")
annotations_path = os.path.join(base_path, "annotations.csv")
output_file = "dataset_cancer.npy"

# --- FONCTIONS ---

def normalize_hu(image):
    """
    Normalise les unités Hounsfield (HU).
    On coupe tout ce qui est < -1000 (air) et > 400 (os).
    Puis on remet entre 0 et 1.
    """
    min_bound = -1000.0
    max_bound = 400.0
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def process_data():
    df = pd.read_csv(annotations_path)
    
    # On filtre pour ne garder que les fichiers présents dans subset0
    subset_files = [f[:-4] for f in os.listdir(subset_path) if f.endswith('.mhd')]
    df_subset = df[df['seriesuid'].isin(subset_files)]
    
    print(f"Traitement de {len(df_subset)} nodules...")
    
    cancer_patches = []
    
    # Boucle sur chaque nodule
    for index, row in tqdm(df_subset.iterrows(), total=df_subset.shape[0]):
        try:
            # 1. Chargement
            img_path = os.path.join(subset_path, row['seriesuid'] + ".mhd")
            itkimage = sitk.ReadImage(img_path)
            origin = np.array(list(reversed(itkimage.GetOrigin())))
            spacing = np.array(list(reversed(itkimage.GetSpacing())))
            ct_scan = sitk.GetArrayFromImage(itkimage)
            
            # 2. Conversion Coordonnées Monde -> Voxel
            world_coord = np.array([row['coordZ'], row['coordY'], row['coordX']])
            voxel_coord = np.absolute(world_coord - origin) / spacing
            voxel_coord = voxel_coord.astype(int)
            
            z, y, x = voxel_coord
            
            # 3. Extraction du Patch (Carré de 50x50 autour du nodule)
            patch_size = 50
            half_size = patch_size // 2
            
            # On normalise l'image entière AVANT de découper
            ct_scan_norm = normalize_hu(ct_scan)
            
            # On extrait la coupe (slice) Z
            img_slice = ct_scan_norm[z]
            
            # On découpe le carré (attention aux bords de l'image)
            # (Si le nodule est trop près du bord, on ignore pour simplifier ce tuto)
            if (y - half_size < 0) or (x - half_size < 0) or \
               (y + half_size > img_slice.shape[0]) or (x + half_size > img_slice.shape[1]):
                continue
                
            patch = img_slice[y-half_size:y+half_size, x-half_size:x+half_size]
            
            cancer_patches.append(patch)
            
        except Exception as e:
            print(f"Erreur sur {row['seriesuid']}: {e}")
            continue

    # Convertir en tableau numpy
    X_data = np.array(cancer_patches)
    
    # Ajouter une dimension pour le channel (nécessaire pour les CNN Keras: 50,50,1)
    X_data = X_data.reshape(X_data.shape[0], 50, 50, 1)
    
    print(f"\nTerminé ! Dataset créé avec {len(X_data)} images de cancer.")
    print(f"Taille du fichier : {X_data.shape}")
    
    # Sauvegarde
    np.save(output_file, X_data)
    print(f"Sauvegardé sous '{output_file}'")

if __name__ == "__main__":
    process_data()