
"""
A module for turning grayscale images into black and white

This module contains a a single class Binarizer which can be used for turning
mnist digit images into black and white instead of grayscale. This helps
when training models.

Classes
-------
- Binarizer

Example
-------
from binarizer import Binarizer
bnz = Binarizer()
X_new = bnz.fit_transform(X)

"""

# Perform necessary imports
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class Binarizer(BaseEstimator,TransformerMixin):
    """
    Transformer that binarizes grayscale images by thresholding pixel intensities.

    Converts pixel values from the range [0, 255] to binary values (0 or 1),
    based on a fixed threshold (default: 64).

    Parameters
    ----------
    None

    Methods
    -------
    fit(X, y=None)
        No fitting required; returns self.
    transform(X)
        Transforms grayscale image data into binary images.
    """
    # Nothing needs to be done in the constructor
    def __init__(self):
        pass
    # Nothing needs to be done for fitting either
    def fit(self,X: np.ndarray,y: np.ndarray = None) -> 'Binarizer':
        return self
    # Change a dark pixel to black and a lighter
    # pixel to white
    def transform(self,X: np.ndarray) -> np.ndarray:
        """
        Transforms grayscale images to binary by applying a threshold of 64.

        Parameters
        ----------
        X : np.ndarray
            Grayscale image array with pixel values ranging from 0 to 255.

        Returns
        -------
        np.ndarray
            Binarized image array with values 0 or 1.
        """
        return np.where(X<64,0,255)
    
