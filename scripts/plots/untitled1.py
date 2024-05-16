"""
Created on Sun May 15 02:20:31 2024

@author: zina
"""
import pandas as pd
from collections import Counter
import spacy
import matplotlib.pyplot as plt

# Charger le modèle linguistique anglais de SpaCy
nlp = spacy.load('en_core_web_sm')

# Charger les données
data_path = '../../data/clean/animal_images_cleaned.csv'
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"File not found: {data_path}")
    exit()

# Fonction pour calculer la longueur moyenne des légendes (en mots)
def average_caption_length(data):
    data['caption_length'] = data['caption'].apply(lambda x: len(x.split()))
    return data['caption_length'].mean()

# Fonction pour calculer la diversité lexicale
def lexical_diversity(data):
    all_words = []
    for doc in nlp.pipe(data['caption'].astype(str)):
        all_words.extend([token.text.lower() for token in doc if not token.is_stop and token.is_alpha])
    lex_div = len(set(all_words)) / len(all_words) if all_words else 0
    return lex_div

# Fonction pour calculer la fréquence des mots sans stopwords
def word_frequency(data):
    all_words = []
    for doc in nlp.pipe(data['caption'].astype(str)):
        all_words.extend([token.text.lower() for token in doc if not token.is_stop and token.is_alpha])
    freq = Counter(all_words)
    return freq.most_common(10)

# Application des calculs
avg_length = average_caption_length(data)
diversity = lexical_diversity(data)
common_words = word_frequency(data)

# Affichage des résultats
print(f"Moyenne de la longueur des légendes: {avg_length:.2f} mots")
print(f"Diversité lexicale: {diversity:.2%}")
print(f"Mots les plus fréquents: {common_words}")

# Visualisations
# 1. Histogramme de la longueur des légendes
plt.figure(figsize=(10, 6))
plt.hist(data['caption_length'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution de la longueur des légendes')
plt.xlabel('Longueur des légendes (en mots)')
plt.ylabel('Fréquence')
plt.grid(True)
plt.savefig('../../plots/caption_length_distribution.png')
plt.show()

# 2. Bar chart des mots les plus fréquents
words, counts = zip(*common_words)
plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue', edgecolor='black')
plt.title('Top 10 des mots les plus fréquents (sans stopwords)')
plt.xlabel('Mots')
plt.ylabel('Fréquence')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig('../../plots/word_frequency.png')
plt.show()

# 3. Diversité lexicale affichée dans un simple texte avec un pie chart pour illustration
labels = 'Diversité lexicale', 'Autres'
sizes = [diversity, 1 - diversity]
colors = ['skyblue', 'lightgrey']
explode = (0.1, 0)  # explode 1st slice

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Diversité lexicale')
plt.savefig('../../plots/lexical_diversity.png')
plt.show()

