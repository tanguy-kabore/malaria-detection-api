# API de Détection du Paludisme

Cette API permet de détecter le paludisme en analysant des images de cellules sanguines. Elle utilise un modèle de segmentation (TernausNet) pour extraire les cellules d'une image et plusieurs modèles de classification pour évaluer si chaque cellule est infectée. De plus, elle génère des visualisations utilisant GradCAM pour montrer où le modèle se concentre lors des prédictions.

## Fonctionnalités
### Fonctionnement de l'API de Détection du Paludisme

L'API de détection du paludisme suit un processus en plusieurs étapes pour analyser les images de frottis sanguins microscopiques. Voici comment elle fonctionne :

1. **Entrée d'Image** : L'utilisateur envoie une image d'un frottis sanguin microscopique à l'API via une requête HTTP.

2. **Segmentation des Cellules** :
   - L'API utilise le modèle de segmentation **TernausNet** pour identifier et extraire les cellules présentes sur le frottis. Ce modèle est spécifiquement conçu pour segmenter des images, permettant de distinguer les cellules sanguines du fond de l'image.
   - Chaque cellule détectée est isolée et préparée pour l'étape suivante.

3. **Classification des Cellules** :
   - Pour chaque cellule segmentée, l'API applique un modèle d'**ensemble EfficientNet-B0**. Ce modèle de classification évalue si la cellule est infectée par le paludisme ou non.
   - Les cellules sont classées en deux catégories : infectées et non infectées.

4. **Analyse des Résultats** :
   - L'API compte le nombre total de cellules détectées et le nombre de cellules infectées.
   - À partir de ces données, elle calcule le **taux d'infection** en pourcentage.

5. **Émission d'un Diagnostic** :
   - En fonction du taux d'infection, l'API génère un **diagnostic** qui indique la probabilité d'infection par le paludisme. Ce diagnostic peut inclure des recommandations pour des tests supplémentaires si nécessaire.

6. **Retour des Résultats** :
   - Les résultats finaux, comprenant le nombre total de cellules, le nombre de cellules infectées, le taux d'infection, et le diagnostic, sont renvoyés à l'utilisateur sous forme d'un objet **JSON**.


## Installation
Pour installer et exécuter cette API, suivez ces étapes :

1. Clonez ce dépôt :
   ```bash
   git clone <URL_DU_DEPOT>
   cd <NOM_DU_DEPOT>
   ```

2. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```

3. Assurez-vous que le modèle TernausNet et les modèles de classification sont présents dans le répertoire approprié :
   ```
   models/
   ├── detection/
   │   ├── model_snapshot_1.h5
   │   ├── model_snapshot_2.h5
   │   ├── model_snapshot_3.h5
   │   ├── model_snapshot_4.h5
   │   └── model_snapshot_5.h5
   └── segmentation/
       └── ternausnet_malaria_model.keras
   ```

4. Exécutez l'API :
   ```bash
   python app.py
   ```

L'API sera accessible à l'adresse `http://127.0.0.1:5000`.

## Utilisation
Pour utiliser l'API, envoyez une requête GET à l'endpoint `/predict` avec le chemin de l'image en tant que paramètre de requête `file`.

### Exemple d'URL
```
http://127.0.0.1:5000/predict?file=chemin/vers/image.jpg
```

## API Endpoints

### `/predict`
- **Méthode** : `GET`
- **Paramètres** :
  - `file` (string) : Chemin d'accès à l'image à analyser.
  
- **Réponse** :
  - Un objet JSON contenant :
    - `total_cells` : Nombre total de cellules détectées.
    - `infected_cells` : Nombre de cellules infectées.
    - `infection_rate` : Taux d'infection.
    - `diagnosis` : Diagnostic basé sur le taux d'infection.
    - `stage` : Stade de l'infection.
    - `gradcam_image_base64` : Image GradCAM encodée en Base64.

## Structure des Données
Les réponses de l'API sont au format JSON, comme suit par exemple:
```json
{
  "total_cells": 10,
  "infected_cells": 3,
  "infection_rate": 0.3,
  "diagnosis": "Paludisme possible, tests supplémentaires recommandés",
  "stage": "stade intermédiaire",
  "gradcam_image_base64": "<image_base64_string>"
}
```

## Dépendances
- Flask
- TensorFlow
- NumPy
- Pillow
- OpenCV
- Matplotlib

Vous pouvez installer toutes les dépendances en utilisant `pip` avec le fichier `requirements.txt`.
