"""
MNIST CNN Training and Evaluation Script.

This script automates the training, evaluation, and saving of a Convolutional 
Neural Network (CNN) classifier for the MNIST dataset. It follows these steps:

1. Fetch Data - Loads the MNIST dataset from disk using fetch().
2. Train Model - Trains the CNN model on the training set (fit()).
3. Refit for Test Evaluation - Retrains the model before evaluation (refit_for_test_set()).
4. Evaluate Model - Computes accuracy ,confidence interval and confusion_matrix (evaluate()).
5. Final Training - Re-trains the model on the full dataset (refit_for_final_model()).
6. Save Model - Saves the trained model to disk (save_model()).

Progress updates are displayed using tqdm.
"""

# Perform necessary imports
from mnist_cnn_classifier import MnistCnnClassifier
from data_fetcher import fetch
from tqdm import tqdm
import os
import time

# Load the dataset
X_train,X_test,X,y_train,y_test,y = fetch(from_disk=False)

# Initialize the classifier
cnn = MnistCnnClassifier()
# Perform the initial fit
tqdm.write('\nInitial fit...\n')
cnn.fit(X_train,y_train)
os.system('cls')
# Re-train for evaluation on the test set
tqdm.write('Refitting for evaluation on test set...\n')
cnn.refit_for_test_set(X_train,y_train)
os.system('cls')
# Perform evaluation
tqdm.write('Performing evaluation...\n')
cnn.evaluate(X_test,y_test)
os.system('cls')
# Re-train on the full dataset
tqdm.write('Refitting for final model...\n')
cnn.refit_for_final_model(X,y)
os.system('cls')
# Save the model
tqdm.write('Saving model...\n')
cnn.save_model()
os.system('cls')
tqdm.write('Done.')