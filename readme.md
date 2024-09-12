# Emotion Recognition Model Using CNN

This project implements an emotion recognition model using Convolutional Neural Networks (CNN). The model is trained on grayscale facial images from the FER dataset to classify facial expressions into one of seven emotions. The system supports modular components for data loading, preprocessing, training, and evaluation.

## Project Structure

- `main.py`: The main script that loads the dataset, preprocesses data, trains the model, and evaluates its performance.
- `data_loader.py`: Handles loading and organizing image data from a specified directory.
- `data_preproocessing.py`: Includes functions for image normalization and label encoding.
- `train.py`: Defines the training process and trains the CNN model on the given data.
- `evaluate.py`: Evaluates the model's performance on test data and visualizes random predictions.
- `utils.py`: Provides helper functions for plotting training history and evaluation results.
- `model.py`: Contains the CNN model architecture used for emotion classification.
- `config.py`:Stores configuration parameters like dataset paths, batch size, epochs, and file paths for saving models and encoders.

## Setup and Installation

Ensure you have Python 3.6+ and TensorFlow installed, then proceed with the following steps to set up the project environment:

1. Install the following Python packages:

    ```bash
    pip install tensorflow pandas matplotlib jiwer
    ```
2. Organize the training and testing datasets in the following structure:
    
    ```bash
    ../Emotions-Analysis/data/train/train/
    ../Emotions-Analysis/data/test/test/
    ```

## Usage

To train and evaluate the model, navigate to the project directory in your terminal and execute:

```bash
python main.py
```

This command will initiate the data loading, preprocessing, and training process. After training, the model weights will be saved in the model.h5 file, and the label encoder will be stored in encoder.pkl.

## Features

1. Data Loading and Preprocessing: The dataset is loaded, shuffled, and the images are normalized for training. Labels are one-hot encoded using a label encoder.
2. CNN Model: The model uses a series of convolutional, pooling, and dropout layers to extract features from 48x48 grayscale facial images and classify them into one of seven emotions.
3. Training and Evaluation: The model is trained with a configurable number of epochs and batch size. It includes plotting of training and validation accuracy and loss graphs.
4. Saved Models and Checkpoints: The model weights are saved after training for future use. A label encoder is also saved to decode predicted labels.
5. Random Prediction Evaluation: After training, the model makes random predictions on test images, displaying the original and predicted labels along with the test image.


## Model Architecture
The CNN model includes:

1. Four convolutional layers with 128, 256, and 512 filters, each followed by max-pooling and dropout layers.
2. Two dense layers for classification with a final softmax layer for the 7 emotion categories.
3. Categorical cross-entropy loss and the Adam optimizer.

## License

Feel free to copy and paste these file structure into your system. This README accurately reflects the setup, dependencies, and usage for the emotions-analysis model .