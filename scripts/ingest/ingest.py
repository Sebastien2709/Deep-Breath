import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CHEMINS (Vérifie bien que tes dossiers s'appellent comme ça) ---
base_path = "."  # Dossier courant
subset_path = os.path.join(base_path, "subset0")
annotations_path = os.path.join(base_path, "annotations.csv")

def load_itk(filename):
    """Charge le scan CT et convertit les distances réelles en pixels"""
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage) # Z, Y, X
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing

def world_to_voxel(world_coord, origin, spacing):
    """Convertit les mm (monde réel) en indices de matrice (pixels)"""
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord.astype(int)

# 1. Vérification des données
if not os.path.exists(subset_path):
    print(f"ERREUR : Le dossier {subset_path} n'existe pas. As-tu bien dézippé subset0.zip ?")
    exit()

print("Chargement des annotations...")
df = pd.read_csv(annotations_path)

# 2. Trouver un patient qui est DANS le subset0
subset_files = [f[:-4] for f in os.listdir(subset_path) if f.endswith('.mhd')]
subset_nodules = df[df['seriesuid'].isin(subset_files)]

print(f"Nombre de nodules trouvés dans subset0 : {len(subset_nodules)}")

if len(subset_nodules) > 0:
    # On prend le 3ème nodule (souvent plus visible que le 1er)
    target = subset_nodules.iloc[2] 
    patient_id = target['seriesuid']
    
    print(f"\nPatient ID : {patient_id}")
    print(f"Localisation tumeur (mm) : {target['coordX']}, {target['coordY']}, {target['coordZ']}")
    
    # 3. Chargement de l'image
    image_path = os.path.join(subset_path, patient_id + ".mhd")
    ct_scan, origin, spacing = load_itk(image_path)
    
    # Conversion mm -> pixel
    coords = np.array([target['coordZ'], target['coordY'], target['coordX']])
    v_coords = world_to_voxel(coords, origin, spacing)
    
    # 4. Affichage
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Vue globale
    z = v_coords[0]
    ax[0].imshow(ct_scan[z], cmap='gray')
    ax[0].set_title(f"Scan Complet (Slice Z={z})")
    # Cercle rouge
    circle = plt.Circle((v_coords[2], v_coords[1]), target['diameter_mm'], color='r', fill=False, linewidth=2)
    ax[0].add_patch(circle)
    
    # Zoom sur le nodule
    pad = 30 # Zoom de 30 pixels autour
    # Attention aux limites de l'image (pour ne pas planter si le nodule est au bord)
    y_min = max(0, v_coords[1]-pad)
    y_max = min(ct_scan.shape[1], v_coords[1]+pad)
    x_min = max(0, v_coords[2]-pad)
    x_max = min(ct_scan.shape[2], v_coords[2]+pad)
    
    patch = ct_scan[z, y_min:y_max, x_min:x_max]
    
    ax[1].imshow(patch, cmap='gray')
    ax[1].set_title("Zoom sur le Nodule (Cancer)")
    
    plt.show()
    
else:
    print("Aucun nodule correspondant trouvé. Vérifie que subset0 contient bien les fichiers .mhd")