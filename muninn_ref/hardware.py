# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact model of Tsetlin inference on a Muninn clause array (Nafarr TsetlinCtrl).

Layout of the clause array (rows x literal columns):

    column 0            valid column: searched with 0, so a row with this include bit
                        set never matches (unused rows, empty clauses)
    columns 1 .. F      features x
    columns F+1 .. 2F   negated features
    further columns     unused (include 0, searched with 1)

    rows                class-major groups of `group` rows; inside a group even rows
                        vote for the class, odd rows against it; rows behind the last
                        group are unused

A search matches a row when no included literal is 0. The class sum is the number of
matching even rows minus matching odd rows of its group; the winner is the class with
the highest sum, the lowest index on ties.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    rows: int
    cols: int                 # literal columns per macro
    classes: int
    group: int = None         # rows per class; default: as many as fit (even)
    tiles: int = 1            # macros side by side

    @property
    def literals(self):
        return self.cols * self.tiles

    @property
    def rows_per_class(self):
        g = self.group or (self.rows // self.classes) // 2 * 2
        if g % 2 or g < 2 or g * self.classes > self.rows:
            raise ValueError(f"{self.classes} classes x {g} rows do not fit {self.rows} rows")
        return g

    def max_features(self):
        return (self.literals - 1) // 2


def _bits(values):
    """Little-endian bit list -> int."""
    out = 0
    for i, v in enumerate(values):
        if v:
            out |= 1 << i
    return out


def export(tm, layout):
    """Clause-array contents (one int per row, bit c = include of column c)."""
    inc = tm.include                                   # (classes, clauses, 2F)
    classes, clauses, lits = inc.shape
    features = lits // 2
    if classes != layout.classes:
        raise ValueError(f"model has {classes} classes, layout {layout.classes}")
    if features > layout.max_features():
        raise ValueError(f"{features} features need {2 * features + 1} literals, "
                         f"layout has {layout.literals}")
    g = layout.rows_per_class
    if clauses > g:
        raise ValueError(f"{clauses} clauses per class, layout has {g} rows per class")
    rows = [1] * layout.rows                           # unused rows: valid bit only
    for k in range(classes):
        for j in range(g):
            if j >= clauses:
                continue                               # padding rows stay unused
            c = inc[k, j]
            if not c.any():
                rows[k * g + j] = 1                    # empty clause outputs 0
                continue
            bits = [False] + list(c[:features]) + list(c[features:])
            rows[k * g + j] = _bits(bits)
    return rows


def literals(x, layout):
    """Search vector for one boolean feature vector."""
    x = np.asarray(x, dtype=bool)
    f = x.size
    bits = [False] + list(x) + list(~x)
    bits += [True] * (layout.literals - len(bits))     # unused columns never block
    if f > layout.max_features():
        raise ValueError("too many features for the layout")
    return _bits(bits)


def search(rows, lit, layout):
    """Row match vector for one search, as the clause array computes it."""
    mask = (1 << layout.literals) - 1
    lit &= ~1                                          # valid column searched with 0
    return [(inc & ~lit & mask) == 0 for inc in rows]


def classify(rows, lit, layout):
    """Class sums and winner, as TsetlinCtrl computes them."""
    matches = search(rows, lit, layout)
    g = layout.rows_per_class
    sums = []
    for k in range(layout.classes):
        group = matches[k * g:(k + 1) * g]
        sums.append(sum(1 if j % 2 == 0 else -1 for j, m in enumerate(group) if m))
    return sums, int(np.argmax(sums))


def predict(rows, X, layout):
    return np.array([classify(rows, literals(x, layout), layout)[1] for x in X])
