import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate_model(model, test_data):
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Test Accuracy: {test_acc}")
    
    Y_pred = model.predict(test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_data.classes
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('results/confusion_matrix.png')
    plt.show()

evaluate_model(model, test_data)
