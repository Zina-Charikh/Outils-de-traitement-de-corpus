import pandas as pd
import numpy as np
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split

# Fixer la seed pour la reproductibilité des résultats
random.seed(42)
np.random.seed(42)

# Chargement des données
data_path = '../../data/clean/animal_images_cleaned.csv'
data = pd.read_csv(data_path)

# Split du corpus en ensembles d'entraînement et de test
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Fonction pour générer des légendes fictives par mélange aléatoire des mots
def generate_fake_captions(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)

# Génération de légendes fictives et calcul des scores BLEU
test['generated_caption'] = test['caption'].apply(generate_fake_captions)
test['bleu_score'] = test.apply(
    lambda row: sentence_bleu([row['caption'].split()], row['generated_caption'].split(), smoothing_function=SmoothingFunction().method1),
    axis=1
)

# Sauvegarde des légendes générées et des scores BLEU dans un fichier CSV
results_path = '../../results/generated_captions_evaluation.csv'
test[['caption', 'generated_caption', 'bleu_score']].to_csv(results_path, index=False)

# Calcul de la moyenne du score BLEU
mean_bleu_score = np.mean(test['bleu_score'])

# Enregistrement des résultats statistiques et explication du score BLEU dans le fichier stats.txt
with open('../../results/evaluation.txt', 'a') as file:
    file.write(f"Moyenne du score BLEU pour l'évaluation: {mean_bleu_score:.4f}\n")
    file.write("Explication du score BLEU :\n")
    file.write("Le score BLEU évalue la qualité des traductions automatiques en comparant les phrases générées avec des références pré-établies sur la base des n-grammes correspondants. ")
    file.write("Il applique une pénalité pour les textes générés courts, ce qui peut conduire à des scores plus bas même lorsque les mots sont corrects mais dans un ordre différent. ")
    file.write("Un score de 1.0 signifie une correspondance parfaite tandis qu'un score de 0.0 indique aucune correspondance. Un score BLEU plus élevé indique une meilleure correspondance avec les textes de référence.\n")

print("Évaluation complète. Résultats et explications sauvegardés dans:", results_path, "et stats.txt")
