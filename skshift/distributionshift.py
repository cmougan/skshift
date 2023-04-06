from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import shap


class ExplanationShiftDetector(BaseEstimator, ClassifierMixin):
    """
    Given a model, and two datasets (source,test), we want to know if the behaviour of the model is different bt train and test.
    We can do this by computing the shap values of the model on the two datasets, and then train a classifier to distinguish between the two datasets.

    Example
    -------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_blobs
    >>> from skshift import ExplanationShiftDetector
    >>> from xgboost import XGBRegressor
    >>> from sklearn.linear_model import LogisticRegression

    >>> X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
    >>> X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

    >>> detector = ExplanationShiftDetector(model=XGBRegressor(),gmodel=LogisticRegression())
    >>> detector.fit(X_tr, y_tr, X_ood)
    >>> detector.get_auc_val()
    # 0.76
    >>> detector.fit(X_tr, y_tr, X_te)
    >>> detector.get_auc_val()
    # 0.5
    """

    def __init__(
        self,
        model,
        gmodel,
        algorithm: str = "auto",
        masker: bool = False,
        data_masker: pd.DataFrame = None,
    ):
        """
        Parameters
        ----------
        model : sklearn model
            Model to be used to compute the shap values.
        gmodel : sklearn model
            Model to be used to distinguish between the two datasets.
        space : str, optional
            Space in which the gmodel is learned. Can be 'explanation' or 'input' or 'predictions'. Default is 'explanation'.

        algorithm : "auto", "permutation", "partition", "tree", or "linear"
                The algorithm used to estimate the Shapley values. There are many different algorithms that
                can be used to estimate the Shapley values (and the related value for constrained games), each
                of these algorithms have various tradeoffs and are preferrable in different situations. By
                default the "auto" options attempts to make the best choice given the passed model and masker,
                but this choice can always be overriden by passing the name of a specific algorithm. The type of
                algorithm used will determine what type of subclass object is returned by this constructor, and
                you can also build those subclasses directly if you prefer or need more fine grained control over
                their options.

        masker : bool,
                The masker object is used to define the background distribution over which the Shapley values
                are estimated. Is a boolean that indicates if the masker should be used or not. If True, the masker is used.
                If False, the masker is not used. The background distribution is the same distribution as we are calculating the Shapley values.
                TODO Decide which masker distribution is better to use, options are: train data, hold out data, ood data
        """

        self.model = model
        self.gmodel = gmodel
        self.explainer = None
        self.algorithm = algorithm
        self.masker = masker
        self.data_masker = data_masker

    def get_gmodel_type(self):
        """
        Returns the type of the gmodel
        """
        if self.gmodel.__class__.__name__ == "Pipeline":
            return self.gmodel.steps[-1][1].__class__.__name__
        else:
            return self.gmodel.__class__.__name__

    def get_model_type(self):
        if self.model.__class__.__name__ == "Pipeline":
            return self.model.steps[-1][1].__class__.__name__
        else:
            return self.model.__class__.__name__

    def fit_detector(self, X, X_ood):
        try:
            check_is_fitted(self.model)
        except:
            raise ValueError(
                "Model is not fitted yet, to use this method the model must be fitted."
            )

        # Get explanations
        S_val = self.get_explanations(X)
        S_ood = self.get_explanations(X_ood)

        # Create dataset for  explanation shift detector
        S_val["label"] = False
        S_ood["label"] = True

        S = pd.concat([S_val, S_ood])
        self.S = S

        self.fit_explanation_shift(S.drop(columns="label"), S["label"])

    def fit_pipeline(self, X, y, X_te, X_ood):
        """
        1. Fits the model F to X and y
        2. Call fit_detector to fit the explanation shift detector
        """

        self.model.fit(X, y)
        self.fit_detector(X_te, X_ood)

    def fit(self, X_source, y_source, X_ood):
        """
        Automatically fits the whole pipeline
        """
        self.fit_pipeline(X_source, y_source, X_source, X_ood)

    def predict(self, X):
        return self.gmodel.predict(self.get_explanations(X))

    def predict_proba(self, X):
        return self.gmodel.predict_proba(self.get_explanations(X))

    def explanation_predict(self, X):
        return self.gmodel.predict(X)

    def explanation_predict_proba(self, X):
        return self.gmodel.predict_proba(X)

    def fit_model(self, X, y):
        self.model.fit(X, y)

    def fit_explanation_shift(self, X, y):
        """
        Fits the explanation shift detector to the data.
        """
        self.gmodel.fit(X, y)

    def get_explanations(self, X):
        """
        Returns the explanations of the model on the data X.
        Produces a dataframe with the explanations of the model on the data X.
        """
        if self.masker:
            self.explainer = shap.Explainer(
                self.model, algorithm=self.algorithm, masker=self.data_masker
            )
        else:
            self.explainer = shap.Explainer(self.model, algorithm=self.algorithm)

        shap_values = self.explainer(X)
        # Name columns
        if isinstance(X, pd.DataFrame):
            columns_name = X.columns
        else:
            columns_name = ["Shap%d" % (i + 1) for i in range(X.shape[1])]

        exp = pd.DataFrame(
            data=shap_values.values,
            columns=columns_name,
        )

        return exp

    def get_auc_val(self):
        """
        Returns the AUC of the explanation shift detector on the validation set of the explanation space
        Example
        -------
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_blobs
        from tools.xaiUtils import ExplanationShiftDetector
        from xgboost import XGBRegressor
        from sklearn.linear_model import LogisticRegression

        # Create data
        X, y = make_blobs(n_samples=2000, centers=2, n_features=5, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        X_ood,y_ood = make_blobs(n_samples=1000, centers=1, n_features=5, random_state=0)

        detector = ExplanationShiftDetector(model=XGBRegressor(),gmodel=LogisticRegression())
        detector.fit(X_tr, y_tr, X_ood)
        detector.get_auc_val()
        # 0.76

        """
        return roc_auc_score(
            self.y_shap_te, self.explanation_predict_proba(self.X_shap_te)[:, 1]
        )

    def get_coefs(self):
        if self.gmodel.__class__.__name__ == "Pipeline":
            if "sklearn.linear_model" in self.gmodel.steps[-1][1].__module__:
                return self.gmodel.steps[-1][1].coef_
            else:
                raise ValueError(
                    "Coefficients can not be calculated. Supported models are linear: sklearn.linear_model, got {}".format(
                        self.gmodel.steps[-1][1].__module__
                    )
                )
        else:
            return self.get_linear_coefs()

    def get_linear_coefs(self):
        if "sklearn.linear_model" in self.gmodel.__module__:
            return self.gmodel.coef_
        else:
            raise ValueError(
                "Coefficients can not be calculated. Supported models are linear: sklearn.linear_model, got {}".format(
                    self.gmodel.steps[-1][1].__module__
                )
            )

    def explain_detector(self):
        exp = shap.Explainer(self.model)

        shap_values = exp.shap_values(self.S_ood.drop(columns="label"))
        shap.summary_plot(
            shap_values, self.S_ood.drop(columns="label"), plot_type="bar", show=False
        )
