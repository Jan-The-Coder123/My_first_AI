from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import os

DATASET_PATH = 'datasets/'
num_classes = len(os.listdir(DATASET_PATH))
class_mode = 'binary' if num_classes == 2 else 'categorical'

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f'{image_path} does not exist')
        return
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f'{image_path} does not funktion')
        return
    model = tf.keras.models.load_model('image_classifier.h5')
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        print(f'{image_path} cant load')
        return

    img = cv2.resize(img, (128, 128))
    img = img / 255
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_names = os.listdir(DATASET_PATH)
    if class_mode == 'binary':
        predicted_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        predicted_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]
    print(f'model prediction: {predicted_class}')
    plt.subplot(1, 1, 1)
    plt.imshow(img_rgb)
    plt.title(f'predicted class: {predicted_class}')
    plt.show()

predict_image('datasets/birds/bird11.png')


