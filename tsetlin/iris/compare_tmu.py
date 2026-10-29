# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: muninn_ref's Tsetlin machine against tmu on the same data and settings."""

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from muninn_ref.booleanize import Thermometer  # noqa: E402
from muninn_ref.tm import TsetlinMachine  # noqa: E402


def main(seeds=range(1, 11), epochs=60, clauses=20, T=10, s=3.0):
    from tmu.tsetlin_machine import TMClassifier
    data = load_iris()
    ours, theirs = [], []
    for seed in seeds:
        Xa, Xb, ya, yb = train_test_split(data.data, data.target, test_size=0.3,
                                          random_state=seed, stratify=data.target)
        th = Thermometer(4, scale=10).fit(Xa)
        Ba, Bb = th.transform(Xa).astype(np.uint32), th.transform(Xb).astype(np.uint32)
        tm = TsetlinMachine(3, clauses, T, s, seed=seed).fit(Ba, ya, epochs)
        ours.append((tm.predict(Bb) == yb).mean())
        np.random.seed(seed)
        ref = TMClassifier(clauses, T, s, platform="CPU")
        for _ in range(epochs):
            ref.fit(Ba, ya)
        theirs.append((ref.predict(Bb) == yb).mean())
    print(f"muninn_ref: mean {np.mean(ours):.3f}  min {np.min(ours):.3f}")
    print(f"tmu:        mean {np.mean(theirs):.3f}  min {np.min(theirs):.3f}")


if __name__ == "__main__":
    main()
