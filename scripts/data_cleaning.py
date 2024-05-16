import os
import pandas as pd
import shutil
import re
from pathlib import Path

def clean_caption(text):
    """ Supprimer les balises HTML du texte """
    return re.sub(r'<[^>]*>', '', text)

def extract_visual_name(url):
    """ Extraire le nom descriptif à partir de l'URL de l'image """
    name = url.split('/')[-1]  # Extrait la dernière partie de l'URL
    name = re.sub(r'\d+', '', name)  # Supprime les chiffres
    name = re.sub(r'\.jpg', '', name, flags=re.IGNORECASE)  # Supprime l'extension .jpg
    name = re.sub(r'[_-]', ' ', name)  # Remplace les tirets et underscores par des espaces
    name = re.sub(r'[^a-zA-Z ]', '', name)  # Supprime la ponctuation restante
    return name.strip()

def main():
    input_dir = '../data/raw'
    output_dir = '../data/clean'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    # Charger le fichier CSV
    df = pd.read_csv(os.path.join(input_dir, 'animal_images_wikimedia.csv'))

    # Nettoyer les données
    df['caption'] = df['caption'].apply(clean_caption)
    df['visual'] = df['visual'].apply(extract_visual_name)
    
    # Copier et renommer les images
    for idx, row in enumerate(df.itertuples(), 1):
        old_path = os.path.join(input_dir, 'images', f"{row.id}.jpg")
        new_name = f"{idx:02d}.jpg"
        new_path = os.path.join(output_dir, 'images', new_name)
        if os.path.isfile(old_path):
            shutil.copy(old_path, new_path)  # Copie au lieu de déplacer

        df.at[row.Index, 'id'] = f"{idx:02d}"
        df.at[row.Index, 'visual'] = row.visual

    # Sauvegarder le nouveau fichier CSV
    df.to_csv(os.path.join(output_dir, 'animal_images_cleaned.csv'), index=False, encoding='utf-8')

    print(f"Le nettoyage est terminé. Les fichiers sont sauvegardés dans {output_dir}.")

if __name__ == "__main__":
    main()
