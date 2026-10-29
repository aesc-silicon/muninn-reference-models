# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from muninn_ref.tm import TsetlinMachine


def noisy_xor(n, rng, noise=0.0, features=12):
    X = rng.integers(0, 2, size=(n, features)).astype(bool)
    y = (X[:, 0] ^ X[:, 1]).astype(int)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return X, y


def test_learns_noisy_xor():
    rng = np.random.default_rng(3)
    X, y = noisy_xor(5000, rng, noise=0.1)       # the classic benchmark setup
    Xt, yt = noisy_xor(2000, rng)
    tm = TsetlinMachine(classes=2, clauses=10, threshold=15, specificity=3.9, seed=3)
    tm.fit(X, y, epochs=5)
    assert (tm.predict(Xt) == yt).mean() > 0.98


def test_rejects_odd_clauses():
    with pytest.raises(ValueError):
        TsetlinMachine(2, 3, 10, 3.0)


def test_empty_clause_outputs_zero_at_inference():
    tm = TsetlinMachine(classes=2, clauses=2, threshold=5, specificity=3.0)
    tm._init(2)
    tm.ta[:] = 1                       # everything excluded
    lit = tm.literals([True, False])
    assert tm.clause_outputs(lit, training=True).all()
    assert not tm.clause_outputs(lit, training=False).any()
