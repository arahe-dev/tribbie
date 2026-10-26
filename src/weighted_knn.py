"""Distance-weighted k-NN with the same inverse-square-root class reweighting as the MoE."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class ClassWeightedKNeighborsClassifier(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights="distance", algorithm="brute", n_jobs=1, class_weight_power=0.5):
        super().__init__(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)
        self.class_weight_power = float(class_weight_power)

    def fit(self, X, y):
        super().fit(X, y)
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        total = float(counts.sum())
        n_classes = len(labels)
        factors = (total / (n_classes * counts.astype(np.float64))) ** self.class_weight_power
        self.class_factors_ = {label: float(factor) for label, factor in zip(labels, factors)}
        return self

    def predict_proba(self, X):
        probabilities = super().predict_proba(X)
        factors = np.asarray([self.class_factors_[label] for label in self.classes_], dtype=np.float64)
        probabilities = probabilities * factors[None, :]
        denominators = probabilities.sum(axis=1, keepdims=True)
        return probabilities / np.maximum(denominators, np.finfo(np.float64).tiny)
