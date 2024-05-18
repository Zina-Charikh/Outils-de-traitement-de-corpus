import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import spacy

# Initialisation de SpaCy pour l'analyse linguistique
nlp = spacy.load('en_core_web_sm')

# Chargement du dataset depuis le disque local
data = load_from_disk('../../dataset/')

# Conversion du dataset en DataFrame pour une manipulation plus facile
df = data.to_pandas()

# Division du dataset en ensembles de train, de validation (dev) et de test
train, temp = train_test_split(df, test_size=0.2, random_state=42)  # 20% pour le test et dev
dev, test = train_test_split(temp, test_size=0.5, random_state=42)  # Divise le reste en deux pour dev et test

# Fonction pour calculer des statistiques détaillées sur les légendes
def calculate_statistics(data):
    data['caption_length'] = data['caption'].apply(lambda x: len(x.split()))
    average_length = np.mean(data['caption_length'])
    std_dev = np.std(data['caption_length'])
    max_length = np.max(data['caption_length'])
    min_length = np.min(data['caption_length'])

    # Calcul de la diversité lexicale
    all_words = []
    for doc in nlp.pipe(data['caption'].astype(str)):
        all_words.extend([token.text.lower() for token in doc if not token.is_stop and token.is_alpha])
    lexical_diversity = len(set(all_words)) / len(all_words) if all_words else 0

    return average_length, std_dev, max_length, min_length, lexical_diversity

# Calcul des statistiques pour l'ensemble de test
avg_length, std_dev, max_length, min_length, lex_div = calculate_statistics(test)

# Sauvegarde des ensembles dans des fichiers CSV
train.to_csv('../../data/train.csv', index=False)
dev.to_csv('../../data/dev.csv', index=False)
test.to_csv('../../data/test.csv', index=False)

# Enregistrement des statistiques dans un fichier texte
with open('../../plots/stats.txt', 'a') as f:
    # Affichage des tailles des ensembles pour confirmation
    f.write(f"Taille de l'ensemble d'entraînement (train): {len(train)}\n")
    f.write(f"Taille de l'ensemble de validation (dev): {len(dev)}\n")
    f.write(f"Taille de l'ensemble de test: {len(test)}\n")
    f.write(f"Moyenne de la longueur des légendes: {avg_length:.2f}\n")
    f.write(f"Écart-type de la longueur des légendes dans l'ensemble de test': {std_dev:.2f}\n")
    f.write(f"Longueur maximale des légendes dans l'ensemble de test: {max_length}\n")
    f.write(f"Longueur minimale des légendes dans l'ensemble de test: {min_length}\n")
    f.write(f"Diversité lexicale dans l'ensemble de test: {lex_div:.2%}\n")

