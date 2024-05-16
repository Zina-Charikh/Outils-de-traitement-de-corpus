#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:22:09 2024

@author: zina
"""

import pandas as pd
import spacy
from collections import Counter
import numpy as np

# Chargement du modèle linguistique SpaCy
nlp = spacy.load('en_core_web_sm')

# Chargement des données
data_path = '../../data/clean/animal_images_cleaned.csv'  # Adaptez le chemin selon votre configuration
data = pd.read_csv(data_path)

# Analyse des légendes
def analyze_captions(data):
    word_lengths = []
    pos_counts = Counter()
    unique_words = set()
    total_words = 0

    for caption in data['caption']:
        doc = nlp(caption)
        word_lengths.extend([len(token.text) for token in doc if token.is_alpha])
        pos_counts.update([token.pos_ for token in doc])
        unique_words.update([token.text.lower() for token in doc if token.is_alpha])
        total_words += len([token.text for token in doc if token.is_alpha])
    
    average_word_length = np.mean(word_lengths)
    diversity_index = len(unique_words) / total_words if total_words > 0 else 0
    pos_distribution = dict(pos_counts)
    
    return average_word_length, diversity_index, pos_distribution

average_word_length, diversity_index, pos_distribution = analyze_captions(data)

print(f"Average Word Length: {average_word_length:.2f}")
print(f"Lexical Diversity Index: {diversity_index:.4f}")
print(f"Part of Speech Distribution: {pos_distribution}")

# Sauvegarde des résultats si nécessaire
result_path = '../../plots/caption_analysis_results.csv'
results = pd.DataFrame([pos_distribution])
results.to_csv(result_path, index=False)
print(f"Results saved to {result_path}")
