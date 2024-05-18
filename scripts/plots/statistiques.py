import spacy
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from scipy.spatial import distance
from datasets import load_from_disk

# Initialisation de SpaCy pour l'analyse linguistique en anglais
nlp = spacy.load('en_core_web_sm')

# Chargement des données depuis un dataset sauvegardé
dataset_path = '../../dataset'  # Assurez-vous que ce chemin est correct
data = load_from_disk(dataset_path)

# Conversion du Dataset Hugging Face en DataFrame pandas
data = data.to_pandas()

def analyser_legends(data):
    mots_tous = []
    longueurs_legende = []
    comptage_pos = Counter()
    mots_uniques = set()
    total_mots = 0

    for legende in data['caption']:
        doc = nlp(legende)
        tokens = [token for token in doc if token.is_alpha and not token.is_stop]
        mots_tous.extend([token.text.lower() for token in tokens])
        longueurs_legende.append(len(legende.split()))
        comptage_pos.update([token.pos_ for token in tokens])
        mots_uniques.update([token.text.lower() for token in tokens])
        total_mots += len(tokens)

    data['longueur_caption'] = longueurs_legende
    moyenne_longueur = np.mean(data['longueur_caption'])
    ecart_type_longueur = np.std(data['longueur_caption'])
    max_longueur = max(data['longueur_caption'])
    min_longueur = min(data['longueur_caption'])
    diversite_lex = len(mots_uniques) / total_mots if total_mots > 0 else 0

    return moyenne_longueur, ecart_type_longueur, max_longueur, min_longueur, diversite_lex, comptage_pos, mots_tous

# Execute analysis function and keep the returned variables
moyenne_longueur, ecart_type_longueur, max_longueur, min_longueur, diversite_lex, distribution_pos, mots = analyser_legends(data)

# Chargement du modèle Word2Vec
word_vectors = api.load("glove-wiki-gigaword-100")

def calculer_similarite(data, mot_cle='animal'):
    similarites = []
    for legende in data['caption']:
        doc = nlp(legende)
        mots_legende = [token.text.lower() for token in doc if token.is_alpha and token.text.lower() in word_vectors.key_to_index]
        distances = [distance.cosine(word_vectors[mot], word_vectors[mot_cle]) for mot in mots_legende if mot in word_vectors]
        similarites.append(np.mean(distances))
    return np.mean(similarites) if similarites else 0

similarite_moyenne = calculer_similarite(data, "animal")

# Enregistrement des résultats
with open('../../plots/stats.txt', 'a') as fichier:
    fichier.write(f"Moyenne de la longueur des légendes: {moyenne_longueur:.2f} mots\n")
    fichier.write(f"Écart-type de la longueur des légendes: {ecart_type_longueur:.2f}\n")
    fichier.write(f"Longueur maximale de légende: {max_longueur}\n")
    fichier.write(f"Longueur minimale de légende: {min_longueur}\n")
    fichier.write(f"Diversité lexicale: {diversite_lex:.2%}\n")
    fichier.write(f"Similarité moyenne avec 'animal': {similarite_moyenne:.4f}\n")
    fichier.write(f"Distribution des parties du discours: {dict(distribution_pos)}\n")

# Visualisation de la distribution de la longueur des légendes
plt.figure(figsize=(10, 6))
plt.hist(data['longueur_caption'], bins=range(0, 160, 10), color='skyblue', edgecolor='black')
plt.title('Distribution de la longueur des légendes')
plt.xlabel('Longueur des légendes (en mots)')
plt.ylabel('Fréquence')
plt.xticks(range(0, 160, 10))
plt.grid(True)
plt.savefig('../../plots/distribution_longueur_legende.png')
plt.close()

# Visualisation de la loi de Zipf
freq_mots = Counter(mots)
frequences = sorted(freq_mots.values(), reverse=True)
plt.figure(figsize=(10, 6))
plt.plot(range(len(frequences)), frequences, linestyle='-', marker='')
plt.title('Loi de Zipf - Distribution des fréquences des mots')
plt.xlabel('Rang du mot')
plt.ylabel('Fréquence')
plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.savefig('../../plots/distribution_zipf.png')
plt.close()

# Visualisation des 50 mots les plus fréquents
mots_frequents = freq_mots.most_common(50)
mots, frequences = zip(*mots_frequents)
plt.figure(figsize=(12, 8))
plt.barh(mots, frequences, color='skyblue')
plt.gca().invert_yaxis()  # Inverser l'axe y pour mettre le mot le plus fréquent en haut
plt.title('Top 50 des mots les plus fréquents')
plt.xlabel('Fréquence')
plt.savefig('../../plots/top_50_words.png')
plt.close()

# Visualisation de la diversité lexicale
labels = 'Diversité lexicale', 'Autres'
sizes = [diversite_lex, 1 - diversite_lex]
colors = ['skyblue', 'lightgrey']
explode = (0.1, 0)
plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Diversité lexicale')
plt.savefig('../../plots/lexical_diversity.png')
plt.close()
