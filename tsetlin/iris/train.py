# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

"""Iris on a Muninn clause array: train, check the hardware reference, export.

    python3 tsetlin/iris/train.py [--macro 64x80] [--seed 1]

The hardware reference (muninn_ref.hardware) must agree with the Tsetlin machine on
every sample; the exported model is what the firmware loads into the macro.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from muninn_ref import export, hardware  # noqa: E402
from muninn_ref.booleanize import Thermometer  # noqa: E402
from muninn_ref.tm import TsetlinMachine  # noqa: E402

HERE = Path(__file__).resolve().parent
MACROS = [(64, 80)]
LABELS = ["setosa", "versicolor", "virginica"]


def dataset(seed=1):
    """Raw (X_train, X_test, y_train, y_test); the split depends on the seed."""
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.3, random_state=seed,
                            stratify=data.target)


def run(macro=(64, 80), seed=1, epochs=60, bits=4, clauses=20, threshold=10, s=3.0,
        out=HERE / "model"):
    X_train, X_test, y_train, y_test = dataset(seed)

    # measurements have one decimal: x10 makes them exact integers for firmware
    therm = Thermometer(bits, scale=10).fit(X_train)
    B_train, B_test = therm.transform(X_train), therm.transform(X_test)

    tm = TsetlinMachine(classes=3, clauses=clauses, threshold=threshold, specificity=s,
                        seed=seed).fit(B_train, y_train, epochs=epochs)
    layout = hardware.Layout(rows=macro[0], cols=macro[1], classes=3)
    rows = hardware.export(tm, layout)

    tm_test = tm.predict(B_test)
    hw_all = hardware.predict(rows, np.vstack([B_train, B_test]), layout)
    tm_all = tm.predict(np.vstack([B_train, B_test]))
    result = {
        "train_accuracy": float((tm.predict(B_train) == y_train).mean()),
        "test_accuracy": float((tm_test == y_test).mean()),
        "hardware_disagreements": int((hw_all != tm_all).sum()),
        "included_literals": int(tm.include.sum()),
        "literals_used": 1 + 2 * B_train.shape[1],
    }
    if out:
        out.mkdir(parents=True, exist_ok=True)
        meta = {"model": "iris", "dataset": "iris (sklearn)", "seed": seed, "epochs": epochs, "bits": bits,
                "clauses_per_class": clauses, "T": threshold, "s": s, **result}
        export.to_json(out / "iris.json", "iris", layout, rows, therm, meta, LABELS)
        export.to_header(out / "iris.h", "iris", layout, rows, therm, LABELS)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--macro", default="64x80")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    r, c = map(int, args.macro.split("x"))
    res = run((r, c), args.seed, args.epochs)
    for k, v in res.items():
        print(f"{k:24} {v}")
    sys.exit(1 if res["hardware_disagreements"] else 0)
