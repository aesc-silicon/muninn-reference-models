# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

"""A vanilla multi-class Tsetlin machine (Granmo, 2018) in NumPy.

Each class owns `clauses` clauses; even-indexed clauses vote for the class, odd ones
against it. A clause is the AND of its included literals; literals are the boolean
features followed by their negations. Every literal of every clause has a Tsetlin
automaton with 2 * states states; it includes the literal in the upper half.

The model is deliberately small and explicit: the trained include matrix is exactly
what the Muninn clause array stores (see muninn_ref.hardware).
"""

import numpy as np


class TsetlinMachine:
    def __init__(self, classes, clauses, threshold, specificity, states=100, seed=0):
        if clauses % 2:
            raise ValueError("clauses per class must be even (half for, half against)")
        self.classes, self.clauses = classes, clauses
        self.T, self.s, self.states = threshold, specificity, states
        self.rng = np.random.default_rng(seed)
        self.ta = None
        self.polarity = np.where(np.arange(clauses) % 2 == 0, 1, -1)

    # ---- model --------------------------------------------------------------
    def _init(self, features):
        self.features = features
        shape = (self.classes, self.clauses, 2 * features)
        # start right at the boundary, half include and half exclude
        self.ta = self.states + self.rng.integers(0, 2, size=shape)

    @property
    def include(self):
        """(classes, clauses, 2 * features) boolean include matrix."""
        return self.ta > self.states

    @staticmethod
    def literals(x):
        x = np.asarray(x, dtype=bool)
        return np.concatenate([x, ~x], axis=-1)

    def clause_outputs(self, lit, training):
        """Clause outputs for one literal vector: (classes, clauses) booleans.

        An empty clause outputs 1 while training and 0 at inference."""
        inc = self.include
        out = ~np.any(inc & ~lit, axis=-1)
        if not training:
            out &= np.any(inc, axis=-1)
        return out

    def class_sums(self, x):
        lit = self.literals(x)
        return (self.clause_outputs(lit, training=False) * self.polarity).sum(axis=-1)

    def predict(self, X):
        return np.array([int(np.argmax(self.class_sums(x))) for x in X])

    # ---- training -----------------------------------------------------------
    def fit(self, X, y, epochs=100):
        X = np.asarray(X, dtype=bool)
        y = np.asarray(y)
        if self.ta is None:
            self._init(X.shape[1])
        for _ in range(epochs):
            for i in self.rng.permutation(len(X)):
                self._update(X[i], y[i])
        return self

    def _update(self, x, target):
        lit = self.literals(x)
        out = self.clause_outputs(lit, training=True)
        other = self.rng.integers(self.classes - 1)
        other += other >= target
        for cls, sign in ((target, 1), (other, -1)):
            v = int(np.clip((out[cls] * self.polarity).sum(), -self.T, self.T))
            p = (self.T - v) / (2 * self.T) if sign > 0 else (self.T + v) / (2 * self.T)
            chosen = self.rng.random(self.clauses) < p
            # target class: positive clauses get Type I, negative Type II;
            # the other class the other way round
            type_i = chosen & ((self.polarity > 0) == (sign > 0))
            type_ii = chosen & ~type_i
            self._type_i(cls, type_i, out[cls], lit)
            self._type_ii(cls, type_ii, out[cls], lit)

    def _type_i(self, cls, mask, out, lit):
        """Recognize patterns: include true literals of firing clauses, forget the rest."""
        if not mask.any():
            return
        ta = self.ta[cls]
        rand = self.rng.random((self.clauses, lit.size))
        fire = (mask & out)[:, None]
        rest = mask[:, None]
        up = fire & lit & (rand < (self.s - 1) / self.s)
        down = rest & ~(fire & lit) & (rand < 1 / self.s)
        ta += up.astype(ta.dtype)
        ta -= down.astype(ta.dtype)
        np.clip(ta, 1, 2 * self.states, out=ta)

    def _type_ii(self, cls, mask, out, lit):
        """Reject false positives: include a false literal into firing clauses."""
        fire = (mask & out)[:, None]
        if not fire.any():
            return
        ta = self.ta[cls]
        ta += (fire & ~lit & (ta <= self.states)).astype(ta.dtype)
