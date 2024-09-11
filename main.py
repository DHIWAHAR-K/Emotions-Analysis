#main.py
import os
import pickle
import config
import numpy as np
from train import train_model
from model import create_model
from utils import plot_history
from evaluate import evaluate_model
from data_loader import load_dataset
from data_preprocessing import extract_features, preprocess_labels

def main():
    # Step 1: Load datasets
    print("Loading datasets...")
    train = load_dataset(config.TRAIN_DIR)
    test = load_dataset(config.TEST_DIR)
    
    # Step 2: Preprocess data
    print("Preprocessing data...")
    x_train = extract_features(train['image'])
    y_train, label_encoder = preprocess_labels(train['label'])
    
    x_test = extract_features(test['image'])
    y_test, _ = preprocess_labels(test['label'])
    
    # Save the label encoder for later evaluation
    with open(config.LABEL_ENCODER_PATH, 'wb') as le_file:
        pickle.dump(label_encoder, le_file)
    
    # Step 3: Check if the model already exists, otherwise train a new one
    if os.path.exists(config.MODEL_PATH):
        print("Loading existing model...")
        model = create_model()
        model.load_weights(config.MODEL_PATH)
    else:
        print("Training the model...")
        history = train_model(x_train, y_train, x_test, y_test)
        
        # Save the model after training
        model.save_weights(config.MODEL_PATH)
        
        # Plot the training history
        plot_history(history)

    # Step 4: Evaluate the model on some test data
    print("Evaluating the model...")
    evaluate_model(model, x_test, y_test, test, label_encoder)

if __name__ == "__main__":
    main()