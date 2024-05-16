import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Chargement des données
data_path = '../../data/clean/animal_images_cleaned.csv'
data = pd.read_csv(data_path)

# Split du corpus en ensembles train, dev, et test
train, temp = train_test_split(data, test_size=0.2, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)

# Sauvegarde des splits en CSV
train.to_csv('../../data/train.csv', index=False)
dev.to_csv('../../data/dev.csv', index=False)
test.to_csv('../../data/test.csv', index=False)

# Ouverture du fichier pour enregistrer les résultats
with open('../../plots/results.txt', 'w') as f:
    f.write(f"Train set size: {len(train)}\n")
    f.write(f"Development set size: {len(dev)}\n")
    f.write(f"Test set size: {len(test)}\n")

    # Génération de la carte du dataset
    # Compter les occurrences de chaque classe si disponible
    if 'label' in data.columns:
        class_counts = data['label'].value_counts()
        plt.figure(figsize=(10, 6))
        class_counts.plot(kind='bar')
        plt.title('Distribution des Classes')
        plt.xlabel('Classe')
        plt.ylabel('Nombre d\'occurrences')
        plt.savefig('../../plots/class_distribution.png')
        plt.close()

    # Calcul de la longueur des légendes et visualisation
    data['caption_length'] = data['caption'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    plt.hist(data['caption_length'], bins=30, color='blue', alpha=0.7)
    plt.title('Distribution de la Longueur des Légendes')
    plt.xlabel('Nombre de mots par légende')
    plt.ylabel('Fréquence')
    plt.savefig('../../plots/length_distribution.png')
    plt.close()

    # Écrire des statistiques descriptives sur la longueur des légendes
    f.write(f"Nombre moyen de mots par légende: {data['caption_length'].mean():.2f}\n")
    f.write(f"Écart-type de la longueur des légendes: {data['caption_length'].std():.2f}\n")
    f.write(f"Longueur maximale de légende: {data['caption_length'].max()}\n")
    f.write(f"Longueur minimale de légende: {data['caption_length'].min()}\n")

print("Les résultats et les cartes du dataset ont été générés et sauvegardés.")
