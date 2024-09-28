# Segmentation Automatique des Cellules Sanguines

## Description

Ce projet se concentre sur la segmentation automatique des cellules sanguines à partir d'images microscopiques, en utilisant le modèle **TernausNet**. La segmentation des cellules est essentielle pour le diagnostic médical, permettant une analyse précise des frottis sanguins.

Le choix de ce modèle repose sur les travaux de **Depto et al. (2021)**, intitulés *Automatic segmentation of blood cells from microscopic slides: a comparative analysis*, publiés dans le journal *Tissue and Cell*. Ce travail met en évidence l'efficacité des modèles de segmentation basés sur les architectures de réseaux de neurones pour cette tâche spécifique.

## Méthodologie

### Choix du Modèle

TernausNet est une architecture de réseau de neurones conçue spécifiquement pour la segmentation d'images. Il s'agit d'une variante de l'architecture U-Net, optimisée pour une meilleure performance en matière de segmentation des structures de petite taille, comme les cellules sanguines. Les caractéristiques de TernausNet, telles que les couches de convolution et les connexions de saut, facilitent la capture de détails fins tout en maintenant une bonne résolution spatiale.

### Données Utilisées

Le dataset utilisé pour l'entraînement et l'évaluation comprend un total de **2 656 images**, dont **1 328 images originales de cellules sanguines** accompagnées de **1 328 mask correspondantes**. Ces images proviennent de frottis sanguins et sont essentielles pour l'entraînement du modèle, car elles permettent au modèle d'apprendre à distinguer les différentes classes de cellules.

- **Lien vers le dataset** : [Télécharger le dataset](https://drive.google.com/file/d/1nG-ra6BPAZSTsdYCvedzCo-JLD7jdH71/view?usp=share_link)

### Résultats d'Entraînement

Les résultats de l'entraînement du modèle sur 10 époques sont les suivants :

| Époque | Précision | Perte   | Précision Validation | Perte Validation |
|--------|-----------|---------|---------------------|------------------|
| 1      | 81.22%    | 0.3907  | 94.87%              | 0.1322           |
| 2      | 95.35%    | 0.1193  | 95.77%              | 0.1101           |
| 3      | 96.61%    | 0.0893  | 96.71%              | 0.0881           |
| 4      | 97.05%    | 0.0780  | 96.75%              | 0.0881           |
| 5      | 97.46%    | 0.0676  | 97.28%              | 0.0736           |
| 6      | 97.63%    | 0.0631  | 97.50%              | 0.0675           |
| 7      | 97.75%    | 0.0596  | 97.57%              | 0.0655           |
| 8      | 97.83%    | 0.0570  | 97.53%              | 0.0661           |
| 9      | 97.77%    | 0.0587  | 97.63%              | 0.0633           |
| 10     | 97.87%    | 0.0560  | 97.39%              | 0.0685           |

Ces résultats démontrent l'efficacité du modèle TernausNet pour la segmentation des cellules sanguines, avec des précisions élevées tant sur l'ensemble d'entraînement que sur l'ensemble de validation.

## Références

Les résultats de ce projet s'appuient sur les travaux suivants :

1. Depto, Deponker Sarker, Rahman, Shazidur, Hosen, Md Mekayel, Akter, Mst Shapna, Reme, Tamanna Rahman, Rahman, Aimon, Zunair, Hasib, Rahman, M. Sohel, Mahdy, MRC. 2021. Automatic segmentation of blood cells from microscopic slides: a comparative analysis. *Tissue and Cell*, 73, 101653. Elsevier. DOI: [Lien vers l'article](https://doi.org/10.1016/j.tice.2021.101653)

2. Blood-cell-segmentation [code github](https://github.com/Deponker/Blood-cell-segmentation/tree/main?tab=readme-ov-file)
