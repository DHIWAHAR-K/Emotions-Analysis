#data_loader.py
import os
import pandas as pd

def load_dataset(directory):
    image_paths = []
    labels = []
    
    for label in os.listdir(directory):
        for filename in os.listdir(directory + label):
            image_path = os.path.join(directory, label, filename)
            image_paths.append(image_path)
            labels.append(label)
            
        print(label, "Completed")
        
    dataset = pd.DataFrame({
        'image': image_paths,
        'label': labels
    })
    return dataset.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset