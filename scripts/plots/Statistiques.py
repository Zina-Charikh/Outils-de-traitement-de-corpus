import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import spacy

# Chargement du modèle linguistique SpaCy pour l'anglais
nlp = spacy.load('en_core_web_sm')

# Chargement des données à partir du fichier CSV
data_path = '../../data/clean/animal_images_cleaned.csv'
data = pd.read_csv(data_path)

# Analyse linguistique pour extraire les fréquences des mots en excluant les stopwords
def process_captions(data):
    all_words = []
    for doc in nlp.pipe(data['caption'].astype(str)):
        all_words.extend([token.text.lower() for token in doc if not token.is_stop and token.is_alpha])
    return all_words

all_words = process_captions(data)
word_freq = Counter(all_words)

# Statistiques descriptives
def compute_statistics(data, word_freq):
    # Longueur moyenne des légendes en mots
    data['caption_length'] = data['caption'].apply(lambda x: len(x.split()))
    avg_length = data['caption_length'].mean()

    # Diversité lexicale
    lex_div = len(set(all_words)) / len(all_words) if all_words else 0

    # Mots les plus fréquents
    common_words = word_freq.most_common(10)

    return avg_length, lex_div, common_words

avg_length, diversity, common_words = compute_statistics(data, word_freq)

# Affichage des résultats calculés
print(f"Moyenne de la longueur des captions: {avg_length:.2f} mots")
print(f"Diversité lexicale: {diversity:.2%}")
print(f"Mots les plus fréquents: {common_words}")

# Visualisation de la distribution de la longueur des légendes
plt.figure(figsize=(10, 6))
plt.hist(data['caption_length'], bins=range(0, 160, 10), color='skyblue', edgecolor='black')
plt.title('Distribution de la longueur des captions')
plt.xlabel('Longueur des captions (en mots)')
plt.ylabel('Fréquence')
plt.xticks(range(0, 160, 10))
plt.grid(True)
plt.savefig('../../plots/caption_length_distribution.png')
plt.show()

# Visualisation de la loi de Zipf
freqs = sorted(word_freq.values(), reverse=True)
plt.figure(figsize=(10, 6))
plt.plot(range(len(freqs)), freqs, linestyle='-', marker='')
plt.title('Loi de Zipf - Distribution des fréquences des mots')
plt.xlabel('Rang du mot')
plt.ylabel('Fréquence')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.savefig('../../plots/zipf_distribution_improved.png')
plt.show()

# Visualisation des mots les plus fréquents
words, counts = zip(*word_freq.most_common(50))
plt.figure(figsize=(12, 8))
plt.barh(range(len(words)), counts, color='skyblue')
plt.yticks(range(len(words)), words)
plt.gca().invert_yaxis()
plt.title('Top 50 des mots les plus fréquents')
plt.xlabel('Fréquence')
plot_path = '../../plots/word_frequency_top_50.png'
plt.savefig(plot_path)
plt.show()

# Visualisation de la diversité lexicale avec un diagramme circulaire
labels = 'Diversité lexicale', 'Autres'
sizes = [diversity, 1 - diversity]
colors = ['skyblue', 'lightgrey']
explode = (0.1, 0)
plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Diversité lexicale')
plt.savefig('../../plots/lexical_diversity.png')
plt.show()
