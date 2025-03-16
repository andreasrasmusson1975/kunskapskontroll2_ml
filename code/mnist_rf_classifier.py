"""
MNIST Random Forest Classifier with Bayesian Optimization.

This module defines `MnistRfClassifier`, a Scikit-Learn-compatible Random Forest classifier
for the MNIST dataset. The model uses `BayesSearchCV` from `scikit-optimize` for 
hyperparameter tuning and supports evaluation with bootstrapped confidence intervals.

Features
--------
* Uses Bayesian Optimization (skopt.BayesSearchCV) for hyperparameter tuning.
* Supports re-training (refit_for_test_set) for final model optimization.
* Computes bootstrapped confidence intervals for accuracy.
* Saves evaluation results, including confusion matrices.
* Implements Scikit-Learn API (fit, predict, evaluate, save_model, load_model).

Usage
-----
>>> from mnist_rf_classifier import MnistRfClassifier
>>> classifier = MnistRfClassifier(n_iter=30)
>>> classifier.fit(X_train, y_train)
>>> preds = classifier.predict(X_test)
>>> classifier.evaluate(X_test, y_test)
>>> classifier.save_model()
"""

# Perform necessary imports
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from elastic_transformer import ElasticTransformer
from tqdm import tqdm
import pandas as pd
from scipy.stats import t
from matplotlib import pyplot as plt
import joblib
from evaluation import Evaluation

class MnistRfClassifier(BaseEstimator):
    """
    A Scikit-Learn-compatible Random Forest classifier for MNIST.

    This class wraps a Random Forest model inside a Scikit-Learn API and 
    optimizes hyperparameters using Bayesian Optimization (BayesSearchCV).

    Attributes
    ----------
    n_iter : int
        Number of iterations for Bayesian hyperparameter optimization.
    pipeline : sklearn.pipeline.Pipeline
        A pipeline containing the Random Forest model.
    space : dict
        The hyperparameter search space for `BayesSearchCV`.
    estimator : sklearn.ensemble.RandomForestClassifier
        The best trained model after Bayesian optimization.
    result_df : pd.DataFrame
        A DataFrame storing the results from Bayesian Optimization.

    Methods
    -------
    fit(X, y)
        Trains the model using Bayesian Optimization.
    predict(X)
        Predicts labels for input data.
    refit_for_test_set(X, y)
        Re-trains the model for test set evaluation.
    refit_for_final_model(X, y)
        Fully re-trains the model as the final version.
    evaluate(X_test, y_test)
        Computes model accuracy, confidence intervals and a confusion matrix.
    save_model()
        Saves the trained model to disk.
    load_model()
        Loads a previously saved model.
    """
    def __init__(self,n_iter: int):
        """
        Initializes the MNIST Random Forest classifier.

        This constructor sets up:
        - A Scikit-Learn `Pipeline` with a `RandomForestClassifier`.
        - A hyperparameter search space for Bayesian Optimization (`BayesSearchCV`).
        - The number of iterations for hyperparameter tuning.

        Parameters
        ----------
        n_iter : int
            The number of iterations for Bayesian hyperparameter search.
        """
        self.n_iter = n_iter
        # Use a pipeline. Unnecessary but mostly for compatibility with the 
        # other classifiers in this package.
        self.pipeline = Pipeline([
            ('rf',RandomForestClassifier(n_jobs=-1))
        ])
        # Define the hyper parameter space
        self.space={
            'rf__n_estimators' : Integer(100,1000),
            'rf__max_depth' : Integer(2,16),
            'rf__min_samples_split' : Integer(2,10),
            'rf__min_samples_leaf' : Integer(1,10),
            'rf__max_features' : Categorical(['sqrt','log2',None]),
            'rf__max_samples' : Real(0.1,1.)
        }
    
    def fit(self,X: np.ndarray,y: np.ndarray) -> 'MnistRfClassifier':
        """
        Trains the Random Forest model using Bayesian Optimization.

        - Uses BayesSearchCV for hyperparameter tuning.
        - Runs the search over a reduced dataset of 5,000 samples for efficiency.
        - Tracks optimization progress with `tqdm`.

        Parameters
        ----------
        X : np.ndarray
            Training images of shape (n_samples, 784) (flattened).
        y : np.ndarray
            Training labels (digits 0-9).

        Returns
        -------
        MnistRfClassifier
            The fitted classifier instance.
        """
        # Reduce the dataset
        X_reduced = X.copy()
        X_reduced = X_reduced[:min(X_reduced.shape[0],5000)]
        y_reduced = y[:min(X_reduced.shape[0],5000)]
        
        # Initialize a progress bar
        pbar = tqdm(total=self.n_iter,)
        # Define a dict to keep track of the best score and parameters
        best_tracker = {"score": -np.inf, "params": None}
        
        # define a callback function for tqdm to update the tracker dictionary
        def tqdm_callback(res):
            current_best_score = -min(res.func_vals)  # Get the best score so far
            if current_best_score > best_tracker["score"]:
                best_tracker["score"] = current_best_score  # Update best score
                best_tracker["params"] = res.x_iters[np.argmax(-res.func_vals)]  # Get best parameters
                pbar.set_postfix({"Best Score": f"{best_tracker['score']:.4f}", "Best Params": best_tracker["params"]})
            
            pbar.update(1)
        # Initialize Bayesian seach cross validation object
        self.opt = BayesSearchCV(
            self.pipeline,
            self.space,
            n_iter = self.n_iter,
        )
        # Fit it
        self.opt.fit(X_reduced,y_reduced,callback = [tqdm_callback])
        # Clear and close the progress bar
        pbar.clear()
        pbar.close()
        # Make sure the best estimator is easily accessible
        self.estimator = self.opt.best_estimator_
        # Create a dataframe with cross validation results and
        # make sure it's easily accessible
        self.result_df = pd.DataFrame(self.opt.cv_results_)
        self.result_df = self.result_df[[
            'param_rf__max_depth',
            'param_rf__max_features',
            'param_rf__max_samples',
            'param_rf__min_samples_leaf',
            'param_rf__min_samples_split',
            'param_rf__n_estimators',
            'mean_test_score',	
            'std_test_score',	
            'rank_test_score'    
        ]]
        return self
    
    def predict(self,X: np.ndarray) -> np.ndarray:
        """
        Predicts digit labels for given input images using the trained Random Forest model.

        Parameters
        ----------
        X : np.ndarray
            Test images of shape (n_samples, 784).

        Returns
        -------
        np.ndarray
            Predicted digit labels.
        """
        return self.estimator.predict(X)
    
    def refit_for_test_set(self,X:np.ndarray,y:np.ndarray) -> 'MnistRfClassifier':
        """
        Refits the model specifically for evaluation on the test set

        Parameters
        ----------
        X : np.ndarray
            Images of shape (n_samples, 784).

        Returns:
        --------
        MnistRfClassifier
            The fitted classifier instance.
        """
        self.estimator.fit(X,y)
        return self
    
    def refit_for_final_model(self,X: np.ndarray,y: np.ndarray) -> 'MnistRfClassifier':
        """
        Refits the model on the full dataset. Mostly a convenience method.

        Parameters
        ----------
        X : np.ndarray
            Images of shape (n_samples, 784).

        Returns:
        --------
        MnistRfClassifier
            The fitted classifier instance.
        """
        self.refit_for_test_set(X,y)
    
    def evaluate(self,X_test: np.ndarray,y_test:np.ndarray):
        """
        Evaluates the Random Forest model on the test set.

        - Computes accuracy and bootstrapped confidence intervals.
        - Generates a confusion matrix.
        - Saves evaluation results to a file.

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
        cm = confusion_matrix(y_test, preds)
        # Create and save an evaluation object
        eval=Evaluation(
            accuracy,
            upper,
            lower,
            cm,
            self.result_df
        )
        joblib.dump(eval,'../evaluations/mnist_rf_evaluation.pkl')

    def save_model(self):
        """
        Saves the trained Random Forest model to disk as '../models/mnist_rf.pkl'.

        Returns
        -------
        None
        """
        joblib.dump(self.estimator,'../models/mnist_rf.pkl')
    
    def load_model(self):
        """
        Loads a previously saved Random Forest model from '../models/mnist_rf.pkl'.

        Returns
        -------
        MnistRfClassifier
            The instance with the loaded model.
        """
        self.estimator = joblib.load('../models/mnist_rf.pkl')
        return self