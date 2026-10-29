<!--
SPDX-FileCopyrightText: 2026 aesc silicon

SPDX-License-Identifier: CC-BY-4.0
-->

# Muninn reference models

Trained models for Muninn compute-in-memory macros, the tools to train and export
them, and a bit-exact Python reference of how the hardware evaluates them.

Each model goes through the same chain, and every step is checked against the
previous one:

1. **Train** in Python (`muninn_ref.tm`, a vanilla Tsetlin machine in NumPy).
2. **Export** to the clause-array layout of a Muninn macro (`muninn_ref.hardware`).
3. **Reference inference** in Python: bit-exact with the Nafarr `TsetlinCtrl`
   frontend on the macro. It must agree with the Tsetlin machine on every sample.
4. **Firmware header** (`muninn_ref.export`): the include bits per row for
   `tsetlin_write_clause()`, plus the booleanizer thresholds.

## Models

| Model | Macro | Classes | Features | Test accuracy | Status |
|---|---|---|---|---|---|
| [iris](tsetlin/iris) | 64 × 80 | 3 | 16 (4 × 4-bit thermometer) | 97.8 % (seed 1) | Python reference |
| [console](tsetlin/console) | 64 × 80 | 5 commands + unknown | 39 hashed bigrams | 95.6 % commands, 82.5 % unknown rejected | Python reference, C booleanizer |
| [console](tsetlin/console) | 2 × 64 × 80, tiled | 5 commands + unknown | 79 hashed bigrams | 96.3 % commands, 92.2 % unknown rejected | Python reference, C booleanizer |

A model depends only on the size of the clause array, not on the layout style: a
64 × 80 model runs on `muninn_tcam_dense_64x80` and `muninn_tcam_conservative_64x80`
alike. Tiled macros sit side by side and AND their match outputs (`tiles` in the
Nafarr `TsetlinCtrl`), so two 64 × 80 macros form 160-literal clauses.

## Usage

The reference host is a container with pinned versions:

```bash
podman build -t muninn-reference-models .
podman run --rm -v $PWD:/work -w /work muninn-reference-models pytest
podman run --rm -v $PWD:/work -w /work muninn-reference-models python3 -m muninn_ref train iris
```

Commands (`python3 -m muninn_ref <command>`):

| Command | Does |
|---|---|
| `list` | available models |
| `train <model> [--macro RxC] [--seed 1] [--epochs N] [-o dir]` | train (every macro of the model unless `--macro`), check against the hardware reference, export JSON and C header |
| `info <model.json>` | layout, rows used, booleanizer, training metadata |
| `check <model.json>` | re-evaluate the exported model on its dataset (no retraining); fails if the accuracy differs from the exported one |
| `classify <model.json> <input…>` | raw input (numbers, or a command line) → booleanizer → clause array → class sums and winner |
| `console <model.json>` | interactive prompt: one input per line, prints class and sums (stdin can be piped) |
| `header <model.json> [-o file]` | regenerate the C header from the JSON |

In Python, `muninn_ref.export.load(path)` returns a `Model` with `layout`, `rows`,
`booleanizer`, `labels`, `reject`, `meta`, and `predict(X)` / `classify(x)` /
`decide(x)` (with the reject rule) on raw inputs.

Without the container: Python ≥ 3.10 with NumPy and scikit-learn. `tmu` is only
needed for the comparison benchmark and requires NumPy 1.x.

## Layout

```
muninn_ref/        Tsetlin machine, booleanizer, hardware reference, export
tsetlin/<model>/   training script, results, exported model (model/*.json, *.h)
docs/              how Tsetlin models map onto Muninn macros, how to train one
tests/             pytest: training, hardware equivalence, exports
```

## License

Code and exported models: Apache-2.0. Documentation: CC-BY-4.0.
