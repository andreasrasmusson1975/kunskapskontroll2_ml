"""
MNIST CNN Classifier with Scikit-Learn Integration

This module defines a convolutional neural network (CNN) classifier for handwritten 
digit recognition using the MNIST dataset. The classifier is built using TensorFlow 
and integrated with Scikit-Learn, allowing it to be used as an sklearn-compatible estimator.

The main class MnistCnnClassifier wraps the CNN in a Scikit-Learn-style API, supporting:
- Model training (fit)
- Predictions (predict)
- Model evaluation (evaluate)
- Saving/loading models (save_model, load_model)

Features
--------
* Uses a CNN model with convolutional layers, batch normalization, dropout, and dense layers.
* Implements preprocessing via `Binarizer` (thresholding) and `CnnTransformer` (reshaping).
* Performs bootstrapped confidence interval estimation for accuracy.
* Saves trained models and evaluation results.

Usage
-----
>>> from mnist_cnn import MnistCnnClassifier
>>> classifier = MnistCnnClassifier()
>>> classifier.fit(X_train, y_train)
>>> preds = classifier.predict(X_test)
>>> classifier.evaluate(X_test, y_test)
>>> classifier.save_model()

"""

# Perform necessary imports
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from scikeras.wrappers import KerasClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from binarizer import Binarizer
from cnn_transformer import CnnTransformer
from scipy.stats import t
from matplotlib import pyplot as plt
import joblib
from evaluation import Evaluation

# Check if gpu is enabled and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('GPU and memory growth enabled')
    except RuntimeError as e:
        pass

def build_model(
    n_neurons1: int = 32, 
    n_neurons2: int = 64, 
    n_neurons3:int = 128,
    n_neurons4: int = 256,
    n_neurons5:int = 512,
    activation: int = 'relu',
    **kwargs
) -> tf.keras.Model:
    """
    Builds a Convolutional Neural Network (CNN) for MNIST classification.

    The network consists of:
    - Convolutional layers with batch normalization.
    - Max pooling for spatial downsampling.
    - Dropout to reduce overfitting.
    - Fully connected layers with ReLU activation.
    - A final softmax layer for digit classification (10 classes).

    Parameters
    ----------
    n_neurons1 : int, optional
        Number of filters in the first convolutional layer (default=32).
    n_neurons2 : int, optional
        Number of filters in the second convolutional layer (default=64).
    n_neurons3 : int, optional
        Number of filters in the third convolutional layer (default=128).
    n_neurons4 : int, optional
        Number of filters in the fourth convolutional layer (default=256).
    n_neurons5 : int, optional
        Number of units in the first fully connected layer (default=512).
    activation : str, optional
        Activation function to use (default='relu').

    Returns
    -------
    tf.keras.Model
        Compiled CNN model.
    """
    # Create a sequential model
    model = tf.keras.Sequential()
    # Add convolutional layers with batch normalization and max pooling
    model.add(tf.keras.layers.Conv2D(n_neurons1, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), use_bias=False,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(n_neurons2, kernel_size=(3,3), activation='relu', use_bias=False,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(n_neurons3, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), use_bias=False,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(n_neurons4, kernel_size=(3,3), activation='relu', use_bias=False,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    # Add dropout layer
    model.add(tf.keras.layers.Dropout(0.25))
    # Flatten for the following dense layer
    model.add(tf.keras.layers.Flatten())
    # Add a dense layer and batch normalization
    model.add(tf.keras.layers.Dense(n_neurons5, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    # Add a dropout layer
    model.add(tf.keras.layers.Dropout(0.5)),
    # Add a final dense layer with softmax activation for the classification
    model.add(tf.keras.layers.Dense(10, activation='softmax', dtype='float32'))
    # Compile the model with appropriat optimizer, loss and metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class MnistCnnClassifier(BaseEstimator):
    """
    A Scikit-Learn-compatible CNN classifier for the MNIST dataset.

    This class wraps a TensorFlow CNN model inside a Scikit-Learn API using BaseEstimator.
    It supports training, prediction, evaluation, and model saving/loading.

    Attributes
    ----------
    estimator : sklearn.pipeline.Pipeline
        A Scikit-Learn pipeline including preprocessing and the CNN model.
    model : tf.keras.Model
        The trained CNN model.

    Methods
    -------
    fit(X, y)
        Trains the CNN model.
    predict(X)
        Predicts digit labels for given input images.
    refit_for_test_set(X, y)
        Re-trains the model specifically for evaluation on the test set.
    refit_for_final_model(X, y)
        Re-trains the model as the final version.
    evaluate(X_test, y_test)
        Evaluates model performance with accuracy and confidence intervals.
    save_model()
        Saves the trained model to disk.
    load_model()
        Loads a saved model from disk.
    """
    def __init__(self):
        pass
    
    def fit(self,X: np.ndarray,y: np.ndarray) -> 'MnistCnnClassifier':
        """
        Trains the CNN model using a Scikit-Learn pipeline.

        The pipeline consists of:
        1. `Binarizer` - Thresholds grayscale images.
        2. `CnnTransformer` - Reshapes images for CNN input.
        3. `KerasClassifier` - The CNN model wrapped as an `sklearn` estimator.

        Parameters
        ----------
        X : np.ndarray
            Training images of shape (n_samples, 784) (flattened).
        y : np.ndarray
            Training labels (digits 0-9).

        Returns
        -------
        MnistCnnClassifier
            The fitted classifier instance.
        """
        # One-hot encode the labels
        y_new = to_categorical(y,num_classes=10)
        # Create a KerasClassifier
        model = KerasClassifier(
            model=build_model, 
            epochs=30, 
            batch_size=32,
            validation_split=0.2,
        )
        # Create the pipeline and fit it
        pipeline = Pipeline([
            ('bnz', Binarizer()),
            ('ctr', CnnTransformer()),
            ('cnn',model)
        ])
        pipeline.fit(X,y_new)
        # Make sure both the pipeline and underlying model are easily accessible
        self.estimator = pipeline
        self.model = self.estimator.named_steps['cnn'].model_
        return self

    def predict(self,X: np.ndarray) -> np.ndarray:
        """
        Predicts digit labels for given input images.

        The input is preprocessed before being passed to the trained CNN model.

        Parameters
        ----------
        X : np.ndarray
            Images of shape (n_samples, 784).

        Returns
        -------
        np.ndarray
            Predicted digit labels.
        """
        # Here, we use separate instances of Binarizer, CnnTransformer and
        # the underlying model instead of the full pipeline. Why? Because
        # we only save the underlying model in h5 format to disk to avoid
        # potential joblib hassles in saving a wrapped tf model. There is
        # no danger of data leakage during training, since the fit method
        # does not use this predict method. Moreover, the Binarizer and
        # CnnTransformer learns nothing from the data.
        bnz = Binarizer()
        X_new = bnz.fit_transform(X)
        ctr = CnnTransformer()
        X_new = ctr.fit_transform(X)
        return np.argmax(self.model.predict(X_new,verbose = 0),axis=1)

    def refit_for_test_set(self,X: np.ndarray,y: np.ndarray) -> 'MnistCnnClassifier':
        """
        Refits the model specifically for evaluation on the test set

        Parameters
        ----------
        X : np.ndarray
            Images of shape (n_samples, 784).

        Returns:
        --------
        MnistCnnClassifier
            The fitted classifier instance.
        """
        # One-hot encode the labels
        y_new = to_categorical(y,num_classes=10)
        # Create a Keras classifier
        model = KerasClassifier(
            model=build_model,
            epochs=30, 
            batch_size=32
        )
        # Create the full pipeline and fit it
        pipeline = Pipeline([
            ('bnz', Binarizer()),
            ('ctr', CnnTransformer()),
            ('cnn',model)
        ])
        pipeline.fit(X,y_new)
        # Make sure both the pipeline and underlying model are easily accessible
        self.estimator=pipeline
        self.model = self.estimator.named_steps['cnn'].model_
        return self
    
    def refit_for_final_model(self,X: np.ndarray,y:np.ndarray) -> 'MnistCnnClassifier':
        """
        Refits the model on the full dataset. Mostly a convenience method.

        Parameters
        ----------
        X : np.ndarray
            Images of shape (n_samples, 784).

        Returns:
        --------
        MnistCnnClassifier
            The fitted classifier instance.
        """
        self.refit_for_test_set(X,y)
        return self

    def evaluate(self,X_test: np.ndarray,y_test: np.ndarray):
        """
        Evaluates the CNN model on the test set.

        - Computes accuracy using bootstrapped sampling.
        - Calculates a 95% confidence interval.
        - Calculates a confusion matrix
        - Generates and saves an evaluation object

        Parameters
        ----------
        X_test : np.ndarray
            Test images.
        y_test : np.ndarray
            True labels.

        Returns
        -------
        None
        """
        # Make predictions and compute the accuracy
        preds = self.predict(X_test)
        accuracy = accuracy_score(y_test,preds)
        accuracies=[]
        # Bootstrap sample the test set and compute upper
        # and lower bounds for a 95 % confidence interval
        # regarding accuracy
        n=300
        for i in tqdm(range(n)):
            sample = np.random.choice(np.arange(y_test.shape[0]),size=y_test.shape[0])
            X_sample,y_sample = X_test[sample,:],y_test[sample]
            preds_sample = self.predict(X_sample)
            accuracies.append(accuracy_score(y_sample,preds_sample))
        accuracies = np.array(accuracies)
        mean_acc = np.mean(accuracies)
        std_err = np.std(accuracies, ddof=1) / np.sqrt(n)
        t_critical = t.ppf((1 + 0.95) / 2, df=n-1)
        margin_of_error = t_critical * std_err
        lower = mean_acc - margin_of_error
        upper = mean_acc + margin_of_error
        # Compute a confusion matrix
        cm = confusion_matrix(y_test, preds)
        # Create and save an evaluation object
        eval=Evaluation(
            accuracy,
            upper,
            lower,
            cm,
            pd.DataFrame()
        )
        joblib.dump(eval,'../evaluations/mnist_cnn_evaluation.pkl')

    def save_model(self):
        """
        Saves the trained CNN model to disk as '../models/mnist_cnn.h5'.

        Returns
        -------
        None
        """
        self.model.save('../models/mnist_cnn.h5')
    def load_model(self):
        """
        Loads a saved CNN model from 'mnist_cnn.h5'.

        Returns
        -------
        MnistCnnClassifier
            The instance with the loaded model.
        """
        self.model = load_model('../models/mnist_cnn.h5')
        return self
        




    