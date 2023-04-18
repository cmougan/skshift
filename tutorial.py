# %%
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from skshift import ExplanationShiftDetector
from xgboost import XGBClassifier
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
# Option 2: fit the whole pipeline of model and detector at once
detector.fit_pipeline(X_tr, y_tr, X_te, X_ood)

print(roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1]))

# %%
# Explaining the change of the model
import shap

explainer = shap.Explainer(detector.detector, masker=detector.get_explanations(X_te))
shap_values = explainer(detector.get_explanations(X_ood_te))
# visualize the first prediction's explanation
shap.waterfall_plot(shap_values[0])
plt.close()

# %%
# Real World Example
from sklearn import datasets

# import some data to play with
dataset = datasets.load_breast_cancer()
X = dataset.data[:, :5]
y = dataset.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)

X_ood = X.copy()
X_ood[:, 0] = X_ood[:, 0] + 3
# Split in train and test
X_ood_tr, X_ood_te, y_ood_tr, y_ood_te = train_test_split(
    X_ood, y, test_size=0.5, random_state=0
)
X_new = np.concatenate([X_te, X_ood_te])
y_new = np.concatenate([np.zeros_like(y_te), np.ones_like(y_ood_te)])

detector = ExplanationShiftDetector(model=XGBClassifier(), gmodel=XGBClassifier())

detector.fit_pipeline(X_tr, y_tr, X_te, X_ood_tr)
roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1])
# %%
explainer = shap.Explainer(detector.detector, masker=detector.get_explanations(X))

shap_values = explainer(detector.get_explanations(X_ood_te))
# Local Explanations
import matplotlib.pyplot as plt

fig = shap.waterfall_plot(shap_values[0], show=False)
plt.savefig("docs/source/images/breastShapLocal.png")
plt.close()
# Global Explanations
fig = shap.plots.bar(shap_values, show=False)
plt.savefig("docs/source/images/breastShapGlobal.png")
plt.close()

# %%
# Folktables
from folktables import ACSDataSource, ACSIncome
import pandas as pd

data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
ca_data = data_source.get_data(states=["CA"], download=True)
pr_data = data_source.get_data(states=["PR"], download=True)
ca_features, ca_labels, _ = ACSIncome.df_to_pandas(ca_data)
pr_features, pr_labels, _ = ACSIncome.df_to_pandas(pr_data)
# %%
# Split ID data and OOD train and test data
X_tr, X_te, y_tr, y_te = train_test_split(
    ca_features, ca_labels, test_size=0.5, random_state=0
)
X_ood_tr, X_ood_te, y_ood_tr, y_ood_te = train_test_split(
    pr_features, pr_labels, test_size=0.5, random_state=0
)
X_new = pd.concat([X_te, X_ood_te])
y_new = np.concatenate([np.zeros_like(y_te), np.ones_like(y_ood_te)])

model = XGBClassifier().fit(X_tr, y_tr)

# %%

detector = ExplanationShiftDetector(model=model, gmodel=XGBClassifier())

detector.fit_detector(X_te, X_ood_te)
print(roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1]))
# 0.96
# %%
explainer = shap.Explainer(detector.detector)

shap_values = explainer(detector.get_explanations(X_new))
# Local Explanations
import matplotlib.pyplot as plt

fig = shap.waterfall_plot(shap_values[0], show=False)
plt.savefig("docs/source/images/folksShapLocal.png")
plt.close()
# Global Explanations
fig = shap.plots.bar(shap_values, show=False)
plt.savefig("docs/source/images/folkstShapGlobal.png")
plt.close()

# %%
# Now if we choose a differet OOD data
tx_data = data_source.get_data(states=["TX"], download=True)
tx_features, tx_labels, _ = ACSIncome.df_to_pandas(tx_data)
# %%
# Split data
X_tr, X_te, y_tr, y_te = train_test_split(
    ca_features, ca_labels, test_size=0.5, random_state=0
)
X_ood_tr, X_ood_te, y_ood_tr, y_ood_te = train_test_split(
    tx_features, tx_labels, test_size=0.5, random_state=0
)
X_new = pd.concat([X_te, X_ood_te])
y_new = np.concatenate([np.zeros_like(y_te), np.ones_like(y_ood_te)])

# %%
detector = ExplanationShiftDetector(model=model, gmodel=XGBClassifier())

detector.fit_detector(X_te, X_ood_te)
print(roc_auc_score(y_new, detector.predict_proba(X_new)[:, 1]))
# 0.82
# %%
explainer = shap.Explainer(detector.detector)

shap_values = explainer(detector.get_explanations(X_new))
# Local Explanations
fig = shap.waterfall_plot(shap_values[0], show=False)
plt.savefig("docs/source/images/folksShapLocalMI.png")
plt.close()
# Global Explanations
fig = shap.plots.bar(shap_values, show=False)
plt.savefig("docs/source/images/folkstShapGlobalMI.png")
plt.close()

# %%
