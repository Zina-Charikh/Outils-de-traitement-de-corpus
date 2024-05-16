# Outils de traitement de corpus

## Tâche choisie : Légendage d'Images (Image-to-Text)
- La tâche consiste à transformer des images en descriptions textuelles qui capturent le contenu et le contexte visuels de l'image.
- **Type de la tâche** : Vision par ordinateur combinée au traitement du langage naturel (NLP).

## Corpus choisi : Textual-Image-Caption Dataset par AhmedSSabir
- Ce corpus est disponible sur Hugging Face et sur le dépôt GitHub de l'auteur : [Visual Semantic Relatedness Dataset for Image Captioning](https://github.com/ahmedssabir/Textual-Visual-Semantic-Dataset).
- **Publication** : Ahmed Sabir, Francesc Moreno-Noguer, Lluís Padró, arXiv, 2023. [Lien vers l'article](https://arxiv.org/pdf/2301.08784).
- **Sources des données** : Le corpus utilise et étend le dataset **COCO Captions**.
- **Taille du corpus** : 59 lignes pour la partie train et 11 lignes pour la partie test.

### Référence bibliographique
```plaintext
@article{sabir2023visual,
  title={Visual Semantic Relatedness Dataset for Image Captioning},
  author={Sabir, Ahmed and Moreno-Noguer, Francesc and Padr{\'o}, Llu{\'\i}s},
  journal={arXiv preprint arXiv:2301.08784},
  year={2023}
}
```
<img src="corpus-ref.png" alt="corpus de référence" />

### Structure du dataset
Le dataset contient trois colonnes :
1. **ID** : Identifiant unique pour chaque image, généralement le nom du fichier.
2. **Visual** : Mots-clés ou tags qui résument les objets ou les concepts présents dans l'image.
3. **Caption** : Description textuelle complète de ce que montre l'image.

### Applications du corpus
Le "Textual-Image-Caption Dataset" peut être utilisé pour :
- **Entraîner** des modèles de légendage d'images.
- **Évaluer** les performances de modèles existants dans la tâche de génération de descriptions textuelles.

### Modèles et architectures
- **Modèle utilisés** : `ydshieh/vit-gpt2-coco-en` sur Hugging Face.
  - **ViT (Vision Transformer)** : Pour la compréhension d'image.
  - **GPT-2** : Pour la génération de texte.
- **Autres modèles** :
  - `OpenAI's CLIP` associé à des modèles de génération de texte comme GPT.
  - `Google's BERT` combiné avec des modèles de vision tels que ResNet.

### Langue et corpus similaires
- **Langue** : Anglais uniquement.
- **Corpus similaires disponibles dans d'autres langues** :
  1. **Flickr30k** : 31,000 images avec cinq légendes en anglais par image. Versions multilingues disponibles.
  2. **Multi30k** : Descriptions en anglais, allemand, français et tchèque.
  3. **STAIR** : Corpus de légendage d'images en japonais.

