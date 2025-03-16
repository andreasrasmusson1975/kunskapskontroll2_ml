"""
A module for reshaping MNIST image data for use in convolutional neural networks (CNNs).

This module provides the `CnnTransformer` class, a scikit-learn-compatible transformer 
that converts flat MNIST image arrays into properly shaped tensors, suitable for CNN input.

Classes
-------
- CnnTransformer

Example
-------
>>> from cnn_transformer import CnnTransformer
>>> ctr = CnnTransformer()
>>> X_new = ctr.fit_transform(X)

"""

# Perform necessary imports
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CnnTransformer(BaseEstimator,TransformerMixin):
    """
    A scikit-learn-compatible transformer that reshapes and scales MNIST image data 
    into a 4D tensor suitable for convolutional neural network input.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer. (No action required; returns self.)
    transform(X)
        Reshapes and scales MNIST data to CNN input format.

    Examples
    --------
    >>> from cnn_transformer import CnnTransformer
    >>> ctr = CnnTransformer()
    >>> X_new = ctr.fit_transform(X)
    """
    
    # Nothing needs to be done in the constructor
    def __init___(self):
        pass
    
    # Nothing needs to be done for fitting either
    def fit(self,X: np.ndarray,y: np.ndarray = None) -> 'CnnTransformer' : 
        return self
    
    def transform(self,X: np.ndarray) -> np.ndarray :
        """
        Reshapes MNIST images from shape (n_samples, 784) to (n_samples, 28, 28, 1) 
        and scales pixel values from 0–255 to 0–1 range.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, 784), containing flattened MNIST images.

        Returns
        -------
        np.ndarray
            Transformed array of shape (n_samples, 28, 28, 1), suitable for CNNs.
        """
        return X.reshape(-1, 28, 28, 1).astype('float32')/255.