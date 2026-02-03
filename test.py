# Python 3
# Import notwendiger Bibliotheken
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Modell laden (MobileNetV2 ist leichtgewichtig und vortrainiert auf ImageNet)
model = MobileNetV2(weights='imagenet')


# Funktion zur Bildvorbereitung und Vorhersage
def erkenne_katze(bildpfad):
    # Bild laden und auf 224x224 Pixel skalieren
    img = Image.open(bildpfad).convert('RGB')
    img = img.resize((224, 224))

    # In numpy Array konvertieren und für das Modell vorbereiten
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Vorhersage
    preds = model.predict(x)
    # Ergebnisse dekodieren (Top 5 Klassen)
    ergebnisse = decode_predictions(preds, top=5)[0]

    # Prüfen, ob eine Katze dabei ist
    katze_klassen = ['n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
                     'n02123478']  # Beispiel ImageNet Katzen-Klassen
    for label_id, name, score in ergebnisse:
        if label_id in katze_klassen:
            return f"Katze erkannt! ({name}, Wahrscheinlichkeit: {score:.2f})"
    return "Keine Katze erkannt."


# Nutzung des Programms
if __name__ == "__main__":
    bildpfad = "katze.jpg"  # Beispielbild
    ergebnis = erkenne_katze(bildpfad)
    print(ergebnis)