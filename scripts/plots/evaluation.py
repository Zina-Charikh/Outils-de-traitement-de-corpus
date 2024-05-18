import random
import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def main():
    random.seed(42)
    np.random.seed(42)

    # Chargement du dataset
    dataset = load_from_disk('../../dataset')

    # Conversion du dataset Hugging Face en DataFrame pandas pour une manipulation plus facile
    df = dataset.to_pandas()

    # Split du corpus en ensembles de train et de test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Génération de légendes fictives et calcul des scores BLEU
    test_df['generated_caption'] = test_df['caption'].apply(generate_fake_captions)
    test_df['bleu_score'] = test_df.apply(
        lambda row: sentence_bleu([row['caption'].split()], row['generated_caption'].split(), smoothing_function=SmoothingFunction().method1),
        axis=1
    )

    # Sauvegarde des légendes générées et des scores BLEU
    results_path = '../../results/generated_captions_evaluation.csv'
    test_df[['caption', 'generated_caption', 'bleu_score']].to_csv(results_path, index=False)

    # Calcul de la moyenne du score BLEU
    mean_bleu_score = np.mean(test_df['bleu_score'])

    # Enregistrement des résultats statistiques et explication du score BLEU
    with open('../../results/evaluation.txt', 'a') as file:
        file.write(f"Moyenne du score BLEU pour l'évaluation: {mean_bleu_score:.4f}\n")
        file.write("Explication du score BLEU :\n")
        file.write("Le score BLEU évalue la qualité des traductions automatiques en comparant les phrases générées avec des références pré-établies sur la base des n-grammes correspondants. ")
        file.write("Il applique une pénalité pour les textes générés courts, ce qui peut conduire à des scores plus bas même lorsque les mots sont corrects mais dans un ordre différent. ")
        file.write("Un score de 1.0 signifie une correspondance parfaite tandis qu'un score de 0.0 indique aucune correspondance. Un score BLEU plus élevé indique une meilleure correspondance avec les textes de référence.\n")

    print("Évaluation complète. Résultats et explications sauvegardés dans:", results_path, "et evaluation.txt")

def generate_fake_captions(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)

if __name__ == "__main__":
    main()
