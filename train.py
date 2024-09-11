#train.py
from model import create_model

def train_model(x_train, y_train, x_test, y_test):
    model = create_model()
    history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))
    return model, history  # Return the model as well