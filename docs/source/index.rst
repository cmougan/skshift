.. image:: https://img.shields.io/pypi/v/skshift.svg
        :target: https://pypi.python.org/pypi/skshift

.. image:: https://github.com/david26694/sktools/workflows/Unit%20Tests/badge.svg
        :target: https://github.com/david26694/sktools/actions

.. image:: https://readthedocs.org/projects/skshift/badge/?version=latest
        :target: https://skshift.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/Workshop-NIPS-blue
        :target: https://img.shields.io/badge/Workshop-NIPS-blue

.. image:: https://static.pepy.tech/personalized-badge/skshift?period=total&units=international_system&left_color=black&right_color=brightgreen&left_text=Downloads
        :target: https://pepy.tech/project/skshift


skshift!
===================================

Sk-shift (scikit-shift) is a Python library designed to detect distribution shift that can affect the behavior of machine learning models.

.. note::

   This project is under active development.

This Python library designed to help users detect distribution shifts that can impact the behavior of machine learning models on new data. 
 
Traditional methods for detecting shifts in input or output data distributions
may have limitations in identifying changes in the model's behavior. 
That's why skshift proposes measures of "explanation shifts" as a key indicator to investigate the interaction between
distribution shifts and learned models.

At its core, skshift defines explanation shift as the statistical comparison between how predictions
from training data are explained and how predictions on new data are explained.
By introducing an Explanation Shift Detector that operates on the explanation space,
skshift provides more sensitive and explainable changes in interactions between distribution shifts and learned models.
Compared to other methods that are based on different kinds of distribution shifts, monitoring for explanation shifts results in better indicators for varying model behavior.

Our approach is backed by both theoretical and experimental evidence,
demonstrating the effectiveness of skshift on synthetic and real data.



Installation
------------

To install skshift, run this command in your terminal:

.. code-block:: console

    $ pip install skshift


Usage: Explanation Shift
-------------------------
To begin detecting shifts in data distributions using skshift, first import the necessary libraries.

.. code:: python

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    from skshift import ExplanationShiftDetector
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import numpy as np



Let's generate synthetic ID and OOD data and split it into train, hold and test sets to avoid overfitting.

.. code:: python

    # Create train, hold and test ID data
    X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    X_hold, y_hold = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)

    # Create OOD data
    X_ood, _ = make_blobs(n_samples=500, centers=1, n_features=5, random_state=0)
    X_ood_te, y_ood_te = make_blobs(n_samples=500, centers=1, n_features=5, random_state=1)

    # Concatenate Distributions
    y_te = np.zeros_like(y_te)
    y_ood_te = np.ones_like(y_ood_te)
    X_new = np.concatenate([X_te, X_ood_te])
    y_new = np.concatenate([y_te, y_ood_te])

Now there is two training options that are equivalent, 
either passing a trained model and just training the Explanation Shift Detector.

Fit Explanation Shift Detector where the classifier is a Gradient Boosting Decision Tree and the Detector a logistic regression. Any other classifier or detector can be used.

.. code:: python

    # Option 1: fit the detector when there is a trained model
    model = XGBClassifier().fit(X_tr, y_tr)

    detector = ExplanationShiftDetector(model=model, gmodel=LogisticRegression())

    detector.fit_detector(X_te, X_ood)
    roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1])
    # 0.7424999999999999


Or fit the whole pipeline without previous retraining.
If the AUC is above 0.5 then we can expect and change on the model predictions.

.. code:: python

    # Option 2: fit the whole pipeline of model and detector at once
    detector = ExplanationShiftDetector(model=XGBClassifier(), gmodel=LogisticRegression())
    detector.fit_pipeline(X_tr, y_tr, X_te, X_ood)

    roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1])
    # 0.7424999999999999

Installation
----------------
.. toctree::
   :maxdepth: 3
   :caption: Getting Started:

   installation


Tutorial
------------
.. toctree::
   :maxdepth: 3
   :caption: Tutorial:

   explanationTutorial



Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: API

   api