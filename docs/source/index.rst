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


Welcome to skshift's documentation!
===================================

sk-shift is a Python library for detecting distribution shift that impacts the model

.. note::

   This project is under active development.

Installation
------------

To install skshift, run this command in your terminal:

.. code-block:: console

    $ pip install skshift


Usage: Explanation Shift
-------------------------
Importing libraries

.. code:: python

    import numpy as np
    from sklearn.datasets import make_blobs
    from xgboost import XGBRegressor
    from sklearn.linear_model import LogisticRegression
    from skshift import ExplanationShiftDetector


Generate synthetic ID and OOD data.

.. code:: python

    X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    X_ood, _ = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

Fit Explanation Shift Detector where the classifier is a Gradient Boosting Decision Tree and the Detector a logistic regression. Any other classifier or detector can be used.

.. code:: python

    detector = ExplanationShiftDetector(model=XGBRegressor(), gmodel=LogisticRegression())
    detector.fit(X, y,X_ood)
   
If the AUC is above 0.5 then we can expect and change on the model predictions.

.. code:: python

    detector.get_auc_val()
    # 0.70



Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
