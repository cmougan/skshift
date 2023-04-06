from skshift import ExplanationShiftDetector

from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import pdb
import pytest
from sklearn.metrics import roc_auc_score

X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
X_hold, y_hold = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)


X_ood, _ = make_blobs(n_samples=500, centers=1, n_features=5, random_state=0)
X_ood_te, y_ood_te = make_blobs(n_samples=500, centers=1, n_features=5, random_state=1)

y_te = np.zeros_like(y_te)
y_ood_te = np.ones_like(y_ood_te)
# concat the two
X_new = np.concatenate([X_te, X_ood_te])
y_new = np.concatenate([y_te, y_ood_te])


def test_fit_detector():
    """
    Test the fit_detector method
    """
    model = XGBClassifier().fit(X_tr, y_tr)

    detector = ExplanationShiftDetector(
        model=model,
        gmodel=LogisticRegression(),
    )

    detector.fit_detector(X_te, X_ood)
    assert (
        np.round_(roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1]), decimals=3)
        == 0.742
    )


def test_ID_data():
    """
    If the X_new is ID then AUC should be 0.5
    """
    model = XGBClassifier().fit(X_tr, y_tr)

    detector = ExplanationShiftDetector(
        model=model,
        gmodel=LogisticRegression(),
    )

    detector.fit_detector(X_te, X_hold)

    y_eval1 = np.zeros_like(y_te)
    y_eval2 = np.ones_like(y_hold)
    y_eval = np.concatenate([y_eval1, y_eval2])
    X_eval = np.concatenate([X_te, X_hold])
    assert (
        np.round_(
            roc_auc_score(y_eval, detector.predict_proba(X_eval)[:, 1]), decimals=1
        )
        == 0.5
    )


def test_fit_detector_logreg():
    """
    Test the fit_detector method with a Log Reg and masked data
    """
    model = LogisticRegression().fit(X_tr, y_tr)

    detector = ExplanationShiftDetector(
        model=model, gmodel=LogisticRegression(), masker=True, data_masker=X_tr
    )

    detector.fit_detector(X_te, X_ood)
    assert np.round_(roc_auc_score(y_new, detector.predict(X_new)), decimals=3) == 0.891


def test_get_explanation_masker():
    """
    Test the get_explanation method with a Linear Regression and masked data
    """
    model = LinearRegression().fit(X_tr, y_tr)

    detector = ExplanationShiftDetector(
        model=model, gmodel=LogisticRegression(), masker=True, data_masker=X_tr
    )

    detector.fit_detector(X_te, X_ood)
    assert roc_auc_score(y_new, detector.predict(X_new)) == 0.907


def test_model_not_fit_error():
    """
    Test the fit_detector method raises an error if the model is not fit
    """
    model = XGBClassifier()
    detector = ExplanationShiftDetector(
        model=model,
        gmodel=LogisticRegression(),
    )
    with pytest.raises(Exception):
        detector.fit_detector(X_te, X_ood)


def fit_pipeline():
    """
    Test that we can fit the full pipeline with the fit_pipeline and fit methods
    """
    detector = ExplanationShiftDetector(
        model=XGBClassifier(),
        gmodel=LogisticRegression(),
    )
    detector.fit_pipeline(X_tr, y_tr, X_te, X_ood)
    assert roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1]) == 0.742
    # Check that it fits the same as the fit method
    detector.fit(X_tr, y_tr, X_te, X_ood)
    assert roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1]) == 0.742
