#data_preprocessing.py
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')  
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features / 255.0  # Normalize the image

def preprocess_labels(labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    return to_categorical(labels_encoded, num_classes=7), le