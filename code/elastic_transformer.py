"""
A module for applying elastic deformations to images, particularly MNIST-style datasets.

This module provides the ElasticTransformer class, a scikit-learn-compatible transformer
that performs elastic deformations on images. Elastic transformations are a type of data 
augmentation that can help neural networks generalize better by introducing small distortions 
similar to natural variations in handwriting.

Classes
-------
- ElasticTransformer

Examples
--------
>>> from elastic_transformer import ElasticTransformer
>>> import numpy as np
>>> X = np.random.rand(100, 784)  # Example flattened MNIST-like data
>>> et = ElasticTransformer(alpha=34, sigma=4, pass_through=False)
>>> X = et.fit_transform(X)
>>> X.shape
(200, 784)
"""

import numpy as np
import scipy.ndimage
from sklearn.base import BaseEstimator,TransformerMixin
from matplotlib import pyplot as plt
import random

class ElasticTransformer(BaseEstimator,TransformerMixin):
    """
    A transformer that applies elastic deformations to images.

    Elastic deformations are created by convolving random displacement fields with 
    a Gaussian filter. This can be useful as a data augmentation technique, 
    particularly for training convolutional neural networks on image datasets.

    Parameters
    ----------
    pass_through : bool, optional (default=False)
        If True, bypasses the transformation and returns the input data unchanged.
    alpha : float, optional (default=34)
        Scaling factor for the displacement fields. Higher values produce stronger distortions.
    sigma : float, optional (default=4)
        Standard deviation for the Gaussian filter used to smooth the displacement fields.

    Methods
    -------
    fit(X, y=None)
        No training required; returns self.
    transform(X)
        Applies elastic deformation to each image in `X`.
    display_example(X)
        Displays a randomly chosen original and transformed image for visualization.

    Examples
    --------
    >>> import numpy as np
    >>> from elastic_transformer import ElasticTransformer
    >>> X = np.random.rand(100, 784)
    >>> et = ElasticTransformer(alpha=34, sigma=4)
    >>> X_transformed = et.fit_transform(X)
    >>> X_transformed.shape
    (200, 784)

    >>> et.display_example(X)
    """
    def __init__(self, pass_through: bool = False,alpha: float = 34.,sigma: float = 4.):
        self.pass_through=pass_through
        self.alpha=alpha
        self.sigma=sigma

    def fit(self,X: np.ndarray, y: np.ndarray = None) -> 'ElasticTransformer':
        return self
    
    def transform(self,X: np.ndarray) -> np.ndarray:
        """
        Applies elastic deformation to a batch of images.

        The transformation uses random displacement fields, smoothed by a Gaussian filter,
        to create localized distortions in the image.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, 784), where each row is a flattened MNIST-like image.

        Returns
        -------
        np.ndarray
            Transformed dataset with elastic deformations. The returned shape is (2 * n_samples, 784),
            as it concatenates the transformed images with the original ones.
        """
        # If pass_through is False, we apply the transformation
        if not self.pass_through:
            X_transformed = X.reshape(-1,28,28)
            X_new=[]
            # Iterate over the observations
            for i in range(X_transformed.shape[0]):
                shape = X_transformed[i].shape
                # Generate random displacement fields in the range [-1, 1], then smooth them with a Gaussian filter.
                # The displacement fields (dx, dy) determine how much each pixel is shifted.
                dx = scipy.ndimage.gaussian_filter((np.random.rand(shape[0],shape[1]) * 2 - 1), self.sigma) * self.alpha
                dy = scipy.ndimage.gaussian_filter((np.random.rand(shape[0],shape[1]) * 2 - 1), self.sigma) * self.alpha
                # Compute new pixel indices after applying the displacement fields (dx, dy).
                # Reshape to match the expected input format of map_coordinates.
                x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
                indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
                # Apply the elastic transformation using map_coordinates, which interpolates pixel values
                # at the new positions defined by indices. The order=1 parameter ensures bilinear interpolation.
                X_new.append(scipy.ndimage.map_coordinates(X_transformed[i], indices, order=1))
            return np.concatenate([np.array(X_new).reshape(X.shape[0],-1),X],axis=0)
        return X

    def display_example(self,X: np.ndarray):
        """
        Displays a randomly selected original and transformed image.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, 784) containing flattened grayscale images.

        Returns
        -------
        None
        """
        idx = np.random.randint(0, X.shape[0])
        X_transformed = self.transform(X[idx].reshape(-1,1)).reshape(-1,28,28)
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(X.reshape(-1,28,28)[idx],cmap='binary')
        axes[1].imshow(X_transformed[0],cmap='binary')
        plt.show()    