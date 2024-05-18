import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
import random
from sklearn.model_selection import train_test_split
from datasets import load_from_disk

# Fixer la seed pour la reproductibilité
random.seed(42)
np.random.seed(42)

# Chargement des données depuis un dataset sauvegardé
dataset_path = '../../dataset'  #
data = load_from_disk(dataset_path)

# Conversion du dataset Hugging Face en DataFrame pandas pour une manipulation plus aisée
data = data.to_pandas()

# Calcul de la longueur de chaque légende
data['longueur_caption'] = data['caption'].apply(lambda x: len(x.split()))

# Élimination des données aberrantes via l'IQR
Q1 = data['longueur_caption'].quantile(0.25)
Q3 = data['longueur_caption'].quantile(0.75)
IQR = Q3 - Q1
data_filtered = data[(data['longueur_caption'] >= (Q1 - 1.5 * IQR)) & (data['longueur_caption'] <= (Q3 + 1.5 * IQR))]

# Split du corpus en train et test
train, _ = train_test_split(data_filtered, test_size=0.2, random_state=42)

# Fonction pour augmenter les légendes par substitution de synonymes
def augment_caption(caption):
    words = caption.split()
    augmented_words = []
    for word in words:
        synonyms = [lem.name() for syn in wordnet.synsets(word) for lem in syn.lemmas() if lem.name() != word]
        augmented_words.append(random.choice(synonyms) if synonyms else word)
    return ' '.join(augmented_words)

# Augmentation des données
train['augmented_caption'] = train['caption'].apply(augment_caption)
train['longueur_augmented'] = train['augmented_caption'].apply(lambda x: len(x.split()))

# Sauvegarde des légendes augmentées
augmented_data_path = '../../results/augmented_captions.csv'
train[['caption', 'augmented_caption']].to_csv(augmented_data_path, index=False)

# Analyse statistique
correlation, p_value = stats.pearsonr(train['longueur_caption'], train['longueur_augmented'])

# Visualisation de la corrélation
plt.scatter(train['longueur_caption'], train['longueur_augmented'], alpha=0.5)
plt.title('Corrélation entre longueur des légendes originales et augmentées')
plt.xlabel('Longueur originale')
plt.ylabel('Longueur augmentée')
plt.grid(True)
plt.savefig('../../results/correlation_augmented_plot.png')

# Enregistrement des résultats statistiques
with open('../../plots/stats.txt', 'a') as fichier:
    fichier.write(f"\nNombre de légendes après élimination des aberrations: {len(data_filtered)}\n")
    fichier.write(f"Corrélation entre légendes originales et augmentées: {correlation}, P-value: {p_value}\n")

