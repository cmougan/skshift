# %%
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from skshift import ExplanationShiftDetector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

# %%
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
# %%
# Option 1: fit the detector when there is a trained model
model = XGBClassifier().fit(X_tr, y_tr)

detector = ExplanationShiftDetector(model=model, gmodel=LogisticRegression())

detector.fit_detector(X_te, X_ood)
print(roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1]))

# %%
# Option 2: fit the whole detector at once
detector.fit_pipeline(X_tr, y_tr, X_te, X_ood)

print(roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1]))
