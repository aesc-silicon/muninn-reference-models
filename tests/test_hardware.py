# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from muninn_ref import hardware
from muninn_ref.tm import TsetlinMachine


def random_tm(classes, clauses, features, rng, empty=0.2):
    tm = TsetlinMachine(classes, clauses, threshold=10, specificity=3.0)
    tm._init(features)
    inc = rng.random((classes, clauses, 2 * features)) < 0.15
    inc[rng.random((classes, clauses)) < empty] = False     # some empty clauses
    tm.ta = np.where(inc, tm.states + 1, tm.states)
    return tm


@pytest.mark.parametrize("classes,clauses,features,layout", [
    (3, 20, 16, hardware.Layout(64, 80, 3)),                # unused rows behind the groups
    (4, 16, 39, hardware.Layout(64, 80, 4)),                # every literal column used
    (4, 10, 30, hardware.Layout(64, 80, 4)),                # padding rows inside groups
    (8, 8, 79, hardware.Layout(64, 160, 8)),
    (2, 32, 60, hardware.Layout(64, 80, 2, tiles=2)),       # two macros side by side
])
def test_hardware_equals_tm(classes, clauses, features, layout):
    rng = np.random.default_rng(features)
    tm = random_tm(classes, clauses, features, rng)
    rows = hardware.export(tm, layout)
    X = rng.integers(0, 2, size=(300, features)).astype(bool)
    assert (hardware.predict(rows, X, layout) == tm.predict(X)).all()
    for x in X[:50]:
        sums, _ = hardware.classify(rows, hardware.literals(x, layout), layout)
        assert sums == list(tm.class_sums(x)[:classes])


def test_semantics_by_hand():
    """A worked example of the clause-array semantics (TsetlinCtrl must match)."""
    layout = hardware.Layout(rows=16, cols=8, classes=2)       # 8 rows per class, 3 features
    rows = [1] * 16
    rows[0] = 0b0000010        # class 0, votes for:     x0
    rows[1] = 0b0100000        # class 0, votes against: not x1
    rows[8] = 0b0001000        # class 1, votes for:     x2
    rows[2] = 0b0000000        # empty row would always match ...
    rows[2] |= 1               # ... so the valid bit blocks it
    lit = hardware.literals([True, True, False], layout)
    # column 0 valid, 1..3 = x0..x2, 4..6 = not x0..x2, column 7 unused (1)
    assert lit == 0b1_100_011_0
    sums, winner = hardware.classify(rows, lit, layout)
    assert sums == [1, 0] and winner == 0
    lit = hardware.literals([False, False, True], layout)
    sums, winner = hardware.classify(rows, lit, layout)
    assert sums == [-1, 1] and winner == 1
    sums, winner = hardware.classify(rows, hardware.literals([False, True, False], layout), layout)
    assert sums == [0, 0] and winner == 0                      # ties: lowest class


def test_layout_limits():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        hardware.export(random_tm(4, 16, 40, rng), hardware.Layout(64, 80, 4))   # 81 literals
    with pytest.raises(ValueError):
        hardware.export(random_tm(4, 18, 8, rng), hardware.Layout(64, 80, 4))    # 18 > 16 rows
    with pytest.raises(ValueError):
        hardware.Layout(16, 8, 9).rows_per_class                                 # no even group
