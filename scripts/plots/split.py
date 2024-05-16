import pandas as pd
from sklearn.model_selection import train_test_split

# Spécifie le chemin vers ton dataset préparé
data_path = '../../data/clean/animal_images_cleaned.csv'  

# Chargement du dataset
data = pd.read_csv(data_path)  

# Division du dataset en ensembles de train, de validation (dev) et de test
train, temp = train_test_split(data, test_size=0.2, random_state=42)  # 20% pour le test et dev
dev, test = train_test_split(temp, test_size=0.5, random_state=42)  # Divise le reste en deux pour dev et test

# Sauvegarde des ensembles dans des fichiers CSV
train.to_csv('../../data/train.csv', index=False)
dev.to_csv('../../data/dev.csv', index=False)
test.to_csv('../../data/test.csv', index=False)

# Affichage des tailles des ensembles pour confirmation
print(f"Taille de l'ensemble d'entraînement (train): {len(train)}")
print(f"Taille de l'ensemble de validation (dev): {len(dev)}")
print(f"Taille de l'ensemble de test: {len(test)}")
