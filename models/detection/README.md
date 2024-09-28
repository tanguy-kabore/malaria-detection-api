
# Détection des Parasites du Paludisme

## Description

Ce projet est inspiré par les travaux de **Mishra, S. (2021)**, intitulé *Malaria Parasite Detection using Efficient Neural Ensembles*, publié dans le *Journal of Electronics, Electromedical Engineering, and Medical Informatics*. L'étude propose une méthode innovante pour la détection des parasites du paludisme à l'aide de techniques d'ensemble basées sur des modèles neuronaux. Le modèle développé ici utilise l'architecture **Snapshot Ensemble** basée sur **EfficientNet-B0**, qui a montré des performances exceptionnelles dans la détection des cellules parasitées.

Le jeu de données utilisé contient **27 558 images** de cellules, équilibrées entre cellules parasitées et non infectées, extraites de frottis sanguins minces. Ce projet vise à automatiser la détection des parasites du paludisme, facilitant ainsi le diagnostic rapide et précis.

## Méthodologie

### Choix de la Méthode

La détection des parasites du paludisme est cruciale pour le diagnostic et le traitement de cette maladie potentiellement mortelle. Les méthodes traditionnelles, fondées sur l'examen microscopique des frottis sanguins, sont longues et nécessitent une expertise spécialisée. 

Pour répondre à ce défi, nous avons opté pour des techniques d'ensemble basées sur l'apprentissage profond. La méthode **Snapshot Ensemble** permet de créer plusieurs instantanés ou apprenants faibles à partir d'un seul réseau neuronal. Cela réduit la nécessité d'entraîner plusieurs modèles distincts tout en maximisant l'efficacité des ressources.

### Modèle Utilisé

Le modèle de base utilisé est **EfficientNet-B0**, pré-entraîné, qui a obtenu les résultats suivants sur le jeu de données :

- **F1 Score** : 99.37%
- **Précision** : 99.52%
- **Rappel** : 99.23%

Ces performances démontrent l'efficacité des modèles d'ensemble, qui exploitent le potentiel prédictif de plusieurs apprenants faibles pour générer un modèle unique capable de gérer des données réelles.

### Visualisation avec GradCAM

L'expérience **GradCAM** a été mise en œuvre pour générer des cartes d'activation du gradient de la dernière couche de convolution. Cela permet de visualiser ce que le modèle observe dans une image pour effectuer des classifications. Cette approche accroît la transparence, l'explicabilité et la fiabilité du modèle, des qualités essentielles pour son déploiement dans le réseau de santé.

## Détails du Dataset

- **Nom** : Dataset Paludisme
- **Taille du téléchargement** : 337.08 MiB
- **Taille de l'ensemble de données** : 317.62 MiB
- **Version** : 1.0.0 (par défaut)
- **Documentation supplémentaire** : [Explorer le code](https://paperswithcode.com/paper/malaria-parasite-detection-using-efficient)
- **Code source** : [tfds.datasets.malaria.Builder](https://lhncbc.nlm.nih.gov/publication/pub9932)

## Reférences

Les résultats de ce projet s'appuient sur les travaux suivants :

1. Mishra, S. 2021. Malaria Parasite Detection using Efficient Neural Ensembles. Journal of Electronics, Electromedical Engineering, and Medical Informatics. 3, 3 (Oct. 2021), 119-133. DOI: [10.35882/jeeemi.v3i3.2](https://doi.org/10.35882/jeeemi.v3i3.2)

2. Malaria-Detection-Using-Deep-Learning-Techniques [code github](https://github.com/sauravmishra1710/Malaria-Detection-Using-Deep-Learning-Techniques/tree/main?tab=readme-ov-file)