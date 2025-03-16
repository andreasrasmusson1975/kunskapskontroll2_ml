"""
A module for evaluating classification model performance.

This module provides the `Evaluation` class, which stores key evaluation metrics 
such as accuracy, confidence intervals, and a confusion matrix. It also includes 
a method for displaying these results.

Classes
-------
- Evaluation

Examples
--------
>>> from evaluation import Evaluation
>>> import numpy as np
>>> from sklearn.metrics import confusion_matrix
>>> y_true = [0, 1, 0, 1, 1]
>>> y_pred = [0, 1, 1, 1, 0]
>>> cm = confusion_matrix(y_true, y_pred)
>>> eval_metrics = Evaluation(accuracy=0.8, upper=0.95, lower=0.65, cm=cm, df=None)
>>> eval_metrics.show()
"""

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import pandas as pd

class Evaluation:
    """
    A class to store and display classification evaluation metrics.

    This class holds classification accuracy, confidence intervals, and a confusion matrix.
    It provides a show() method to display these results in a structured manner.

    Parameters
    ----------
    accuracy : float
        The point estimate for classification accuracy on the test set.
    upper : float
        The upper bound of the 95% confidence interval for accuracy.
    lower : float
        The lower bound of the 95% confidence interval for accuracy.
    cm : np.ndarray
        The confusion matrix of the classification results.
    df : optional
        An optional DataFrame containing additional evaluation details.

    Methods
    -------
    show()
        Prints the accuracy and confidence interval, then displays the confusion matrix.

    Examples
    --------
    >>> from evaluation import Evaluation
    >>> import numpy as np
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [0, 1, 0, 1, 1]
    >>> y_pred = [0, 1, 1, 1, 0]
    >>> cm = confusion_matrix(y_true, y_pred)
    >>> eval_metrics = Evaluation(accuracy=0.8, upper=0.95, lower=0.65, cm=cm, df=None)
    >>> eval_metrics.show()
    """
    def __init__(
            self,
            accuracy: float,
            upper: float,
            lower: float,
            cm: np.ndarray
            ,df: pd.DataFrame
    ):
        self.accuracy=accuracy
        self.upper = upper
        self.lower=lower
        self.cm=cm
        self.df=df
    
    def show(self):
        """
        Displays the accuracy, confidence interval, and confusion matrix.

        Prints the accuracy with its 95% confidence interval and plots the confusion matrix.

        Returns
        -------
        None
        """
        print(f'Point estimation for accuracy (test set): {self.accuracy}')
        print(f'95 % confidence interval for accuracy: ({self.lower},{self.upper})')
        ConfusionMatrixDisplay(self.cm).plot()
        plt.show()