# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

"""Thermometer encoding: bit k of a feature is 1 when the value exceeds threshold k.

Thresholds are quantiles of the training data, so every bit splits the data at a
meaningful point. Firmware applies the same comparisons, which is all the
booleanizer front end of the Tsetlin engine needs to do.
"""

import numpy as np


class Thermometer:
    """`scale` > 0 quantizes: values and thresholds become integers (value * scale),
    exactly as firmware compares them."""

    def __init__(self, bits, scale=None):
        self.bits, self.scale = bits, scale
        self.thresholds = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        q = np.linspace(0, 1, self.bits + 2)[1:-1]
        t = np.quantile(X, q, axis=0).T                        # (features, bits)
        self.thresholds = np.round(t * self.scale).astype(int) if self.scale else t
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.scale:
            X = np.round(X * self.scale).astype(int)
        return (X[:, :, None] > self.thresholds[None, :, :]).reshape(len(X), -1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
