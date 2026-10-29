# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

"""Command line for the reference models.

    python3 -m muninn_ref list
    python3 -m muninn_ref train iris [--macro 64x80] [--seed 1] [--epochs 60] [-o dir]
    python3 -m muninn_ref train console                 # console_64x80 and console_64x160
    python3 -m muninn_ref info tsetlin/iris/model/iris.json
    python3 -m muninn_ref check tsetlin/iris/model/iris.json
    python3 -m muninn_ref classify tsetlin/iris/model/iris.json 5.1 3.5 1.4 0.2
    python3 -m muninn_ref header tsetlin/iris/model/iris.json [-o iris.h]
    python3 -m muninn_ref console tsetlin/iris/model/iris.json
    python3 -m muninn_ref console tsetlin/console/model/console_64x160.json

`check` re-evaluates an exported model on its dataset with the hardware reference,
without retraining, and compares the accuracies with the ones recorded at export.
"""

import argparse
import importlib
import sys
from pathlib import Path

from muninn_ref import export

ROOT = Path(__file__).resolve().parents[1]


def models():
    return sorted(p.parent.name for p in (ROOT / "tsetlin").glob("*/train.py"))


def trainer(name):
    if name not in models():
        sys.exit(f"unknown model '{name}', available: {', '.join(models())}")
    return importlib.import_module(f"tsetlin.{name}.train")


def cmd_list(args):
    for m in models():
        print(m)


def cmd_train(args):
    mod = trainer(args.model)
    macros = [tuple(map(int, args.macro.split("x")))] if args.macro else mod.MACROS
    kw = {"out": Path(args.out)} if args.out else {}
    if args.epochs:
        kw["epochs"] = args.epochs
    fail = 0
    for r, c in macros:
        if len(macros) > 1:
            print(f"--- {args.model} {r}x{c}")
        res = mod.run((r, c), args.seed, **kw)
        for k, v in res.items():
            print(f"{k:24} {v}")
        fail |= bool(res["hardware_disagreements"])
    return int(fail)


def cmd_info(args):
    m = export.load(args.json)
    lay = m.layout
    used = sum(1 for r in m.rows if r != 1)
    print(f"{'name':24} {m.name}")
    print(f"{'macro':24} {lay.rows}x{lay.cols}" + (f" x{lay.tiles} tiles" if lay.tiles > 1 else ""))
    print(f"{'classes':24} {lay.classes} x {lay.rows_per_class} rows")
    print(f"{'rows used':24} {used} of {lay.rows}")
    print(f"{'included literals':24} {sum(bin(r >> 1).count('1') for r in m.rows)}")
    if m.text:
        print(f"{'booleanizer':24} {m.booleanizer.features} hashed character bigrams")
    elif m.booleanizer is not None:
        f, b = m.booleanizer.thresholds.shape
        print(f"{'booleanizer':24} {f} features x {b}-bit thermometer, scale {m.booleanizer.scale}")
    if m.labels:
        print(f"{'labels':24} {', '.join(m.labels)}")
    if m.reject:
        print(f"{'reject':24} {m.reject}")
    for k, v in m.meta.items():
        print(f"{k:24} {v}")


def cmd_check(args):
    m = export.load(args.json)
    mod = trainer(m.meta.get("model", m.name))
    args_ = [m.meta.get("seed", 1)] + m.meta.get("samples", [])
    X_train, X_test, y_train, y_test = mod.dataset(*args_)
    got = {"train_accuracy": float((m.predict(X_train) == y_train).mean()),
           "test_accuracy": float((m.predict(X_test) == y_test).mean())}
    fail = False
    for k, v in got.items():
        want = m.meta.get(k)
        ok = want is None or abs(v - want) < 1e-9
        fail |= not ok
        print(f"{k:24} {v:.4f}" + ("" if want is None else f"   exported {want:.4f}")
              + ("" if ok else "   MISMATCH"))
    return 1 if fail else 0


def _input(m, words):
    """Command-line words or a console line -> raw model input."""
    if m.text:
        return " ".join(words)
    f = m.booleanizer.thresholds.shape[0] if m.booleanizer is not None else None
    values = [float(v) for v in " ".join(words).replace(",", " ").split()]
    if f and len(values) != f:
        raise ValueError(f"expected {f} values, got {len(values)}")
    return values


def _result(m, x):
    k, sums = m.decide(x)
    what = f"class {k} ({m.label(k)})" if m.labels else f"class {k}"
    if k is None:
        what = "unknown"
    return f"{what}  sums {' '.join(f'{s:+d}' for s in sums)}"


def cmd_classify(args):
    m = export.load(args.json)
    print(_result(m, _input(m, args.values)))


def cmd_console(args):
    """Line-based prompt, as the firmware demo will run it on the UART."""
    m = export.load(args.json)
    interactive = sys.stdin.isatty()
    if interactive:
        what = "a command" if m.text else "values separated by spaces, 'q'"
        print(f"muninn {m.name}: enter {what}, Ctrl-D to quit")
    while True:
        if interactive:
            print("> ", end="", flush=True)
        line = sys.stdin.readline()
        if not line or (not m.text and line.strip() == "q"):
            break
        if not line.strip():
            continue
        try:
            print(_result(m, _input(m, [line.strip()])))
        except ValueError as e:
            print(f"error: {e}")


def cmd_header(args):
    m = export.load(args.json)
    out = Path(args.output or Path(args.json).with_suffix(".h"))
    export.to_header(out, m.name, m.layout, m.rows, m.booleanizer, m.labels, m.reject)
    print(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="muninn_ref", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="available models")
    p = sub.add_parser("train", help="train a model and export it")
    p.add_argument("model")
    p.add_argument("--macro", help="e.g. 64x80 (default: every macro of the model)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, help="default: the model's")
    p.add_argument("-o", "--out", help="output directory (default: tsetlin/<model>/model)")
    for name, hlp in (("info", "summary of an exported model"),
                      ("check", "re-evaluate an exported model without retraining")):
        sub.add_parser(name, help=hlp).add_argument("json")
    p = sub.add_parser("classify", help="classify one raw input")
    p.add_argument("json")
    p.add_argument("values", nargs="+", help="numbers, or the words of a command")
    sub.add_parser("console", help="interactive prompt: one input per line").add_argument("json")
    p = sub.add_parser("header", help="regenerate the C header from the JSON")
    p.add_argument("json")
    p.add_argument("-o", "--output")
    args = ap.parse_args(argv)
    return globals()[f"cmd_{args.cmd}"](args) or 0


if __name__ == "__main__":
    sys.exit(main())
