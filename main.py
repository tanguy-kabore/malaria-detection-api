from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import io
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration CORS pour permettre uniquement les origines de confiance
CORS(app, resources={r"/predict": {"origins": ["https://malaria-detection-app.vercel.app"]}})

# Définition d'une classe Dropout fixe personnalisée
class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)

# Enregistrement de la couche Dropout fixe personnalisée
get_custom_objects().update({'FixedDropout': FixedDropout})

# Fonction pour charger le modèle TernausNet pour la segmentation
def load_ternausnet_model():
    model = load_model('models/segmentation/datas/ternausnet_malaria_model.keras')
    return model

# Fonction pour charger les modèles de classification pré-entraînés
def load_snapshot_models():
    with tf.keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
        models = [
            load_model('models/detection/datas/model_snapshot_1.h5'),
            load_model('models/detection/datas/model_snapshot_2.h5'),
            load_model('models/detection/datas/model_snapshot_3.h5'),
            load_model('models/detection/datas/model_snapshot_4.h5'),
            load_model('models/detection/datas/model_snapshot_5.h5')
        ]
    return models

# Chargement des modèles
segmentation_model = load_ternausnet_model()
classification_models = load_snapshot_models()

# Prétraitement de l'image d'entrée
def preprocess_image(image, target_size):
    image = image.resize(target_size)  # Redimensionner l'image
    image = np.array(image) / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajouter la dimension de lot
    return image

# Prédiction d'ensemble
def ensemble_predictions(models, cell_img):
    predictions = []
    for model in models:
        input_shape = model.input_shape[1:3]
        processed_image = preprocess_image(cell_img, target_size=input_shape)
        predictions.append(model.predict(processed_image))
    return np.mean(predictions, axis=0)

# Implémentation de GradCAM
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# Fonction pour superposer la carte de chaleur sur l'image originale
def overlay_gradcam_on_image(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    return np.uint8(superimposed_img)

# Fonction pour sauvegarder la visualisation GradCAM
def save_gradcam_visual(cell_img, model, cell_img_np):
    heatmap = generate_gradcam_heatmap(model, cell_img_np, last_conv_layer_name='block7a_expand_conv')
    superimposed_img = overlay_gradcam_on_image(np.array(cell_img), heatmap)
    
    # Sauvegarde de l'image GradCAM
    gradcam_img = Image.fromarray(superimposed_img)
    img_io = io.BytesIO()
    gradcam_img.save(img_io, format='PNG')
    img_io.seek(0)
    return img_io

# Fonction de segmentation utilisant TernausNet
def segment_cells_with_ternausnet(image):
    input_shape = segmentation_model.input_shape[1:3]
    processed_image = preprocess_image(image, target_size=input_shape)
    
    mask = segmentation_model.predict(processed_image)[0]
    mask = np.squeeze(mask)
    mask = (mask > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cell_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cell = np.array(image)[y:y+h, x:x+w]
        cell_pil = Image.fromarray(cell)
        cell_images.append(cell_pil)
    
    return cell_images

def classify_cells(cell_images):
    infected_count = 0
    total_cells = len(cell_images)
    gradcam_images = []
    total_confidence = 0  # Variable pour accumuler les probabilités de prédiction

    for cell_img in cell_images:
        # Prétraitement de l'image de la cellule
        cell_img_np = preprocess_image(cell_img, target_size=classification_models[0].input_shape[1:3])
        
        # Obtenir la prédiction sous forme de probabilité (valeur entre 0 et 1)
        prediction = ensemble_predictions(classification_models, cell_img)
        
        # Ajout de la probabilité de prédiction pour calculer la fiabilité globale
        total_confidence += prediction
        
        # Si la probabilité est supérieure à 0.5, la cellule est infectée
        if prediction > 0.5:
            infected_count += 1

        # Génération de la visualisation GradCAM pour la cellule actuelle
        gradcam_img_io = save_gradcam_visual(cell_img, classification_models[0], cell_img_np)
        gradcam_images.append(gradcam_img_io)

    # Calcul du taux de fiabilité moyen : somme des probabilités de prédiction / nombre total de cellules
    average_confidence = total_confidence / total_cells if total_cells > 0 else 0

    return total_cells, infected_count, average_confidence, gradcam_images

# Fonction pour combiner plusieurs images en une seule grille
def combine_images(image_list, img_size=(64, 64)):  # Réduction de la taille des images à 64x64
    num_images = len(image_list)
    
    # Calcul automatique de la taille de la grille
    grid_w = int(np.ceil(np.sqrt(num_images)))  # Largeur de la grille
    grid_h = int(np.ceil(num_images / grid_w))  # Hauteur de la grille

    fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * img_size[0] // 100, grid_h * img_size[1] // 100))
    
    for i, ax in enumerate(axes.flat):
        if i < len(image_list):
            img = Image.open(image_list[i]) if isinstance(image_list[i], io.BytesIO) else image_list[i]
            ax.imshow(img.resize(img_size))
        ax.axis('off')

    # Sauvegarde de la grille en image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='PNG')
    buf.seek(0)
    plt.close(fig)

    return buf

# Route pour gérer les prédictions à partir d'un fichier envoyé dans la requête
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    image = Image.open(file.stream)  # Ouvrir le fichier d'image

    cell_images = segment_cells_with_ternausnet(image)

    total_cells, infected_cells, average_confidence, gradcam_images = classify_cells(cell_images)

    # Combinaison de toutes les images GradCAM en une seule grille d'images
    combined_gradcam_img = combine_images([io.BytesIO(img.getbuffer()) for img in gradcam_images])

    # Combinaison de toutes les cellules détectées en une seule grille d'images
    combined_cells_img = combine_images(cell_images)

    # Encodage des images en Base64
    gradcam_base64 = base64.b64encode(combined_gradcam_img.getvalue()).decode('utf-8')
    cells_base64 = base64.b64encode(combined_cells_img.getvalue()).decode('utf-8')

    # Logique de décision basée sur le pourcentage de cellules infectées
    if total_cells > 0:
        if infected_cells > 0:
            result = "Infection détectée"
        else:
            result = "Pas d'infection détectée"
    else:
        result = "Aucune cellule détectée"

    return jsonify({
        'total_cells': total_cells,
        'infected_cells': infected_cells,
        'average_confidence': float(average_confidence),
        'result': result,
        'gradcam_image': gradcam_base64,
        'cells_image': cells_base64
    })

# Route pour sauvegarder les annotations
@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    # Récupérer le corps de la requête
    data = request.json
    
    # Vérifiez que les données contiennent un fichier et des annotations
    if 'file' not in data or 'annotations' not in data:
        return jsonify({'message': 'Fichier ou annotations manquants'}), 400

    # Extraire les données
    file_data = data['file']
    annotations = data['annotations']

    # Enlevez le préfixe de la chaîne base64 (data:image/jpeg;base64,)
    header, encoded = file_data.split(",", 1)

    # Création des dossiers s'ils n'existent pas
    parasitized_dir = 'datasets/Parasitized'
    uninfected_dir = 'datasets/Uninfected'
    if not os.path.exists(parasitized_dir):
        os.makedirs(parasitized_dir)
    if not os.path.exists(uninfected_dir):
        os.makedirs(uninfected_dir)

    # Décoder les données de l'image originale
    original_image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    width, height = original_image.size

    # Boucle sur les annotations pour enregistrer les images
    for index, annotation in enumerate(annotations):
        # Vérifiez que les clés nécessaires existent
        if 'x' not in annotation or 'y' not in annotation or 'w' not in annotation or 'h' not in annotation:
            return jsonify({'message': 'Clés manquantes dans les annotations'}), 400

        # Convertir les coordonnées en pixels
        x = int(annotation['x'] * width)
        y = int(annotation['y'] * height)
        w = int(annotation['w'] * width)
        h = int(annotation['h'] * height)

        # Vérifiez le label pour définir le chemin de sauvegarde
        label = annotation.get('cls', 'uninfected')  # Utilisez 'uninfected' par défaut si le label n'est pas fourni
        if label.lower() == 'parasitized':
            save_path = os.path.join(parasitized_dir, f'cell_{index}.jpg')
        else:
            save_path = os.path.join(uninfected_dir, f'cell_{index}.jpg')

        # Extraire la zone d'intérêt de l'image originale
        cell_image = original_image.crop((x, y, x + w, y + h))

        # Sauvegarder l'image
        cell_image.save(save_path)

    return jsonify({'message': 'Annotations et images sauvegardées avec succès'}), 200

if __name__ == '__main__':
    app.run(debug=True)