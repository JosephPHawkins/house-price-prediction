import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from functools import partial

# --- Addition ---
def column_addition(X):
    return X.sum(axis=1).reshape(-1, 1)

def addition_name(_, __):
    return ['addition']

def addition_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_addition, feature_names_out=addition_name)
    )

# --- Multiplication ---
def column_multiplication(X):
    return X.prod(axis=1).reshape(-1, 1)

def multiplication_name(_, __):
    return ['multiplication']

def multiplication_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_multiplication, feature_names_out=multiplication_name)
    )

# --- Subtraction ---
def column_subtract_all(X):
    return (X[:, 0] - X[:, 1:].sum(axis=1)).reshape(-1, 1)

def subtraction_name(_, __):
    return ['subtraction']

def subtraction_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_subtract_all, feature_names_out=subtraction_name)
    )

# --- Weighted Bathrooms ---
def weighted_bathrooms_func(X):
    return (X[:, 0] + 0.5 * X[:, 1] + X[:, 2] + 0.5 * X[:, 3]).reshape(-1, 1)

def weighted_bathrooms_name(_, __):
    return ['weighted_bathrooms']

def weighted_bathrooms_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(weighted_bathrooms_func, feature_names_out=weighted_bathrooms_name)
    )

# --- RBF Kernel Transformer ---
class RBFKernelSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, target_year=1960, gamma=0.1):
        self.target_year = target_year
        self.gamma = gamma

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        if hasattr(X, "values"):
            X = X.values
        return rbf_kernel(X.reshape(-1, 1), [[self.target_year]], gamma=self.gamma)

def rbf_feature_namer(_, feature_names_in, target_year=1960, name='rbf_similarity'):
    return [f"{name}_{target_year}"]

def rbf_kernel_pipeline(target_year, gamma=0.1, name='rbf_similarity'):
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(
            func=RBFKernelSimilarity(target_year, gamma).transform,
            feature_names_out=partial(rbf_feature_namer, target_year=target_year, name=name)
        )
    )