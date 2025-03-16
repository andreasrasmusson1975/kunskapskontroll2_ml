"""
A module for loading and preprocessing the MNIST dataset with elastic augmentation.

This module provides a single function fetch, which retrieves the MNIST dataset from OpenML,
applies elastic transformations to augment the training data, and saves the processed datasets to disk for quick subsequent loading.

Functions
---------
fetch(from_disk=False)
    Loads MNIST data, applies augmentation, splits the data, and optionally caches results.

Examples
--------
>>> from mnist_loader import fetch
>>> X_train, X_test, X, y_train, y_test, y = fetch()
>>> X_train.shape
(120000, 784)

>>> # Loading directly from saved files (faster)
>>> X_train, X_test, X, y_train, y_test, y = fetch(from_disk=True)
"""

import numpy as np
from elastic_transformer import ElasticTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.base import BaseEstimator, TransformerMixin

def fetch(from_disk: bool = False) -> tuple:
    """
    Load and preprocess the MNIST dataset with optional disk caching.

    Fetches the MNIST dataset, applies elastic transformations to augment the training dataset,
    and shuffles all datasets. Processed datasets are optionally saved as npy files.

    Parameters
    ----------
    from_disk : bool, default=False
        If True, load the processed datasets directly from cached npy files.

    Returns
    -------
    tuple
        A tuple containing:
        - X_train : np.ndarray, training images (augmented).
        - X_test : np.ndarray, test images (not augmented).
        - X : np.ndarray, all images (augmented).
        - y_train : np.ndarray, labels for training images.
        - y_test : np.ndarray, labels for test images.
        - y : np.ndarray, labels for all images.
    """
    if not from_disk:
        # Load from openml
        mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False,parser='auto')
        X = mnist["data"][:10000]
        y = mnist["target"].astype(np.uint8)[:10000]

        # Split into train and test data
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1000,shuffle=True,random_state=42)

        # Apply elastic transformations to the training data
        et = ElasticTransformer()
        X_train = et.fit_transform(X_train)
        y_train = np.concatenate([y_train,y_train],axis=0)
        X=et.transform(X)
        y=np.concatenate([y,y],axis=0)

        # Shuffle the training data
        rng = np.random.default_rng()
        permutations = rng.permutation(X_train.shape[0])
        X_train=X_train[permutations,:]
        y_train=y_train[permutations]

        # Shuffle all data
        permutations = rng.permutation(X.shape[0])
        X=X[permutations,:]
        y=y[permutations]

        # Save the result to disk
        np.save('../saved_datasets/X_train.npy',X_train)
        np.save('../saved_datasets/X_test.npy',X_test)
        np.save('../saved_datasets/X.npy',X)
        np.save('../saved_datasets/y_train.npy',y_train)
        np.save('../saved_datasets/y_test.npy',y_test)
        np.save('../saved_datasets/y.npy',y)

        print(X_train.shape,X_test.shape)
        return X_train,X_test,X,y_train,y_test,y
    
    # Load data from disk
    X_train=np.load('../saved_datasets/X_train.npy')
    X_test=np.load('../saved_datasets/X_test.npy')
    X=np.load('../saved_datasets/X.npy')
    y_train=np.load('../saved_datasets/y_train.npy')
    y_test=np.load('../saved_datasets/y_test.npy')
    y=np.load('../saved_datasets/y.npy')
    return X_train,X_test,X,y_train,y_test,y