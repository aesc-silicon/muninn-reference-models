# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

"""Console commands with typos on a Muninn clause array.

    python3 -m muninn_ref train console [--macro 64x80]

Classes are the commands of the ElemRV bare-metal shell plus "unknown". Training data
are the commands with up to two typos and, for "unknown", other shell words and random
letters that are more than two edits away from every command. One model per macro:
console_64x80 (39 features) and console_64x160 (79 features).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from muninn_ref import export, hardware  # noqa: E402
from muninn_ref.text import Bigrams, distance, typo  # noqa: E402
from muninn_ref.tm import TsetlinMachine  # noqa: E402

HERE = Path(__file__).resolve().parent
MACROS = [(64, 80), (64, 160)]
COMMANDS = ["help", "version", "led on", "led off", "clear"]
LABELS = COMMANDS + ["unknown"]
UNKNOWN = len(COMMANDS)
MAX_TYPOS = 2
OTHER = ("ls cd cat exit reboot reset status make vi git run start stop gpio uart led blink "
         "on off info dump read write test hello quit list show set get time date mem peek "
         "poke boot load flash erase ping echo top kill").split()
LETTERS = list("abcdefghijklmnopqrstuvwxyz  ")


def _unknown(rng):
    while True:
        r = rng.random()
        if r < 0.4:
            w = rng.choice(OTHER)
        elif r < 0.6:
            w = f"{rng.choice(OTHER)} {rng.choice(OTHER)}"
        else:
            w = "".join(rng.choice(LETTERS, rng.integers(1, 10))).strip()
        if w and all(distance(w, c) > MAX_TYPOS for c in COMMANDS):
            return str(w)


def _samples(rng, n):
    X, y = [], []
    for _ in range(n):
        k = int(rng.integers(len(LABELS)))
        if k == UNKNOWN:
            X.append(_unknown(rng))
        else:
            w = COMMANDS[k]
            for _ in range(rng.choice([0, 1, 1, 2])):
                w = typo(w, rng)
            X.append(str(w))
        y.append(k)
    return X, np.array(y)


def dataset(seed=1, train=3000, test=2000):
    """(X_train, X_test, y_train, y_test): lines of text and class indices."""
    rng = np.random.default_rng(seed)
    X_train, y_train = _samples(rng, train)
    X_test, y_test = _samples(rng, test)
    return X_train, X_test, y_train, y_test


def run(macro=(64, 160), seed=1, epochs=100, clauses=10, threshold=6, s=1.5, min_sum=1,
        out=HERE / "model", samples=(3000, 2000)):
    X_train, X_test, y_train, y_test = dataset(seed, *samples)
    layout = hardware.Layout(rows=macro[0], cols=macro[1], classes=len(LABELS))
    bigrams = Bigrams(layout.max_features())
    B_train, B_test = bigrams.transform(X_train), bigrams.transform(X_test)

    tm = TsetlinMachine(classes=len(LABELS), clauses=clauses, threshold=threshold,
                        specificity=s, seed=seed).fit(B_train, y_train, epochs=epochs)
    rows = hardware.export(tm, layout)
    reject = {"class": UNKNOWN, "min_sum": min_sum}
    name = f"console_{macro[0]}x{macro[1]}"
    model = export.Model(name, layout, rows, bigrams, labels=LABELS, reject=reject)

    tm_test = tm.predict(B_test)
    hw_all = hardware.predict(rows, np.vstack([B_train, B_test]), layout)
    decided = np.array([UNKNOWN if k is None else k
                        for k, _ in (model.decide(x) for x in X_test)])
    cmd = y_test != UNKNOWN
    result = {
        "train_accuracy": float((tm.predict(B_train) == y_train).mean()),
        "test_accuracy": float((tm_test == y_test).mean()),
        "commands_recognized": float((decided[cmd] == y_test[cmd]).mean()),
        "unknown_rejected": float((decided[~cmd] == UNKNOWN).mean()),
        "exact_commands": int(sum(model.decide(c)[0] == k for k, c in enumerate(COMMANDS))),
        "hardware_disagreements": int((hw_all != tm.predict(np.vstack([B_train, B_test]))).sum()),
        "included_literals": int(tm.include.sum()),
        "literals_used": 1 + 2 * bigrams.features,
    }
    if out:
        out.mkdir(parents=True, exist_ok=True)
        meta = {"model": "console", "dataset": "synthetic typos", "seed": seed,
                "samples": list(samples), "max_typos": MAX_TYPOS, "epochs": epochs,
                "clauses_per_class": clauses, "T": threshold, "s": s, **result}
        export.to_json(out / f"{name}.json", name, layout, rows, bigrams, meta, LABELS, reject)
        export.to_header(out / f"{name}.h", name, layout, rows, bigrams, LABELS, reject)
    return result
