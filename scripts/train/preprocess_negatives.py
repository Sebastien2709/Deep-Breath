import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm

# --- CONFIG ---
base_path = "."
subset_path = os.path.join(base_path, "subset0")
candidates_path = os.path.join(base_path, "candidates.csv")
output_file = "dataset_negative.npy"

# Nombre de négatifs qu'on veut récupérer (pour équilibrer avec tes ~112 positifs)
NUM_NEGATIVES = 200 

def normalize_hu(image):
    min_bound = -1000.0
    max_bound = 400.0
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def process_negatives():
    # 1. Lecture du CSV
    print("Lecture de candidates.csv...")
    df = pd.read_csv(candidates_path)
    
    # 2. Filtres
    # On veut seulement la classe 0 (pas cancer)
    df = df[df['class'] == 0]
    
    # On ne garde que les fichiers présents dans subset0
    subset_files = [f[:-4] for f in os.listdir(subset_path) if f.endswith('.mhd')]
    df_subset = df[df['seriesuid'].isin(subset_files)]
    
    print(f"Candidats négatifs disponibles dans subset0 : {len(df_subset)}")
    
    # 3. Échantillonnage aléatoire
    # On mélange et on en prend juste un petit paquet pour aller vite
    df_subset = df_subset.sample(n=NUM_NEGATIVES, random_state=42)
    print(f"Sélection de {NUM_NEGATIVES} candidats pour l'entraînement...")

    negative_patches = []
    
    for index, row in tqdm(df_subset.iterrows(), total=df_subset.shape[0]):
        try:
            img_path = os.path.join(subset_path, row['seriesuid'] + ".mhd")
            
            itkimage = sitk.ReadImage(img_path)
            origin = np.array(list(reversed(itkimage.GetOrigin())))
            spacing = np.array(list(reversed(itkimage.GetSpacing())))
            ct_scan = sitk.GetArrayFromImage(itkimage)
            
            world_coord = np.array([row['coordZ'], row['coordY'], row['coordX']])
            voxel_coord = np.absolute(world_coord - origin) / spacing
            voxel_coord = voxel_coord.astype(int)
            
            z, y, x = voxel_coord
            
            patch_size = 50
            half_size = patch_size // 2
            
            # Normalisation et Extraction
            ct_scan_norm = normalize_hu(ct_scan)
            
            # Vérification des bords
            if z < 0 or z >= ct_scan_norm.shape[0]: continue
            
            img_slice = ct_scan_norm[z]
            
            if (y - half_size < 0) or (x - half_size < 0) or \
               (y + half_size > img_slice.shape[0]) or (x + half_size > img_slice.shape[1]):
                continue
                
            patch = img_slice[y-half_size:y+half_size, x-half_size:x+half_size]
            negative_patches.append(patch)
            
        except Exception as e:
            continue

    # Conversion Numpy et Reshape
    X_neg = np.array(negative_patches)
    X_neg = X_neg.reshape(X_neg.shape[0], 50, 50, 1)
    
    print(f"\nTerminé ! Dataset Négatif créé.")
    print(f"Taille : {X_neg.shape}")
    
    np.save(output_file, X_neg)
    print(f"Sauvegardé sous '{output_file}'")

if __name__ == "__main__":
    process_negatives()