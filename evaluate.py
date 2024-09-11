#evaluate.py
import random
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, x_test, y_test, test, label_encoder):
    image_index = random.randint(0, len(test) - 1)
    print("Original Output:", test['label'][image_index])
    pred = model.predict(x_test[image_index].reshape(1, 48, 48, 1))
    prediction_label = label_encoder.inverse_transform([pred.argmax()])[0]
    print("Predicted Output:", prediction_label)
    plt.imshow(x_test[image_index].reshape(48, 48), cmap='gray')
    plt.show()