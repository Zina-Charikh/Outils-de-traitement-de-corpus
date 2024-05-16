#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:23:54 2024

@author: zina
"""
import spacy
import numpy as np
import pandas as pd
import gensim.downloader as api
from scipy.spatial import distance

nlp = spacy.load('en_core_web_sm')


# Chargement des données
data_path = '../../data/clean/animal_images_cleaned.csv'  # Adaptez le chemin selon votre configuration
data = pd.read_csv(data_path)

# Charger un modèle Word2Vec pré-entraîné
word_vectors = api.load("glove-wiki-gigaword-100")

# Fonction pour calculer la similarité moyenne des mots dans les légendes par rapport à un mot clé
def caption_similarity(data, keyword):
    similarities = []
    
    for caption in data['caption']:
        doc = nlp(caption)
        words = [token.text.lower() for token in doc if token.is_alpha and token.text.lower() in word_vectors.key_to_index]
        if words:
            word_distances = [distance.cosine(word_vectors[w], word_vectors[keyword]) for w in words if w in word_vectors]
            if word_distances:
                similarities.append(np.mean(word_distances))
    
    return np.mean(similarities) if similarities else 0

# Calculer la similarité par rapport à un mot clé pertinent
keyword = 'animal'  # Changez selon le contexte de votre corpus
mean_similarity = caption_similarity(data, keyword)

print(f"Mean Semantic Similarity to '{keyword}': {mean_similarity:.4f}")
