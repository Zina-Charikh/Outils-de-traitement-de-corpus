#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 22:24:35 2024

@author: zina
"""
import requests
from bs4 import BeautifulSoup
import os
import csv
import spacy


nlp = spacy.load('en_core_web_sm')

# URL de la source des données
base_url = 'https://commons.wikimedia.org'
search_url = 'https://commons.wikimedia.org/w/index.php?search=Animal&title=Special:MediaSearch&type=image&uselang=fr'
response = requests.get(search_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Dossier pour enregistrer les images et le fichier CSV
raw_data_path = 'data/raw'
os.makedirs(raw_data_path, exist_ok=True)

# Fichier CSV pour les légendes
csv_file_path = os.path.join(raw_data_path, 'images_captions.csv')
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'visual', 'caption'])  # En-tête du CSV

    # Extraction et sauvegarde des images et des légendes
    for img in soup.find_all('img'):
        src = img['src']
        caption = img.get('alt', 'No caption available')
        if src.startswith('/'):
            src = base_url + src
        
        img_response = requests.get(src)
        if img_response.status_code == 200:
            img_name = os.path.basename(src)  # Nom de fichier de l'image
            img_path = os.path.join(raw_data_path, img_name)
            with open(img_path, 'wb') as f:
                f.write(img_response.content)
            
            # Analyser la légende pour extraire les noms communs (NOUN)
            doc = nlp(caption)
            visual = ', '.join([token.text for token in doc if token.pos_ == 'NOUN'])
            
            writer.writerow([img_name, visual, caption])
