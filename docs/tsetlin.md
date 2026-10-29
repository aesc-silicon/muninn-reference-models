<!--
SPDX-FileCopyrightText: 2026 aesc silicon

SPDX-License-Identifier: CC-BY-4.0
-->

# Tsetlin machines on Muninn clause arrays

## The model

A Tsetlin machine classifies boolean feature vectors. Each class owns a set of
clauses. A clause is the AND of the literals it includes, where the literals are the
features x and their negations ¬x. Half of a class's clauses vote for it, half
against it. The class sum is (votes for) − (votes against), and the class with the
highest sum wins.

Training decides, for every clause and literal, whether to include it. After
training only these include bits remain. They are the whole model.

## Mapping onto a clause array

A Muninn `muninn_tcam` macro stores one include bit per cell. A search checks all
rows at once: a row matches when none of its included literals is 0.

| Clause array | Tsetlin machine |
|---|---|
| row | one clause |
| column 0 | valid column: searched with 0; set in unused rows and empty clauses |
| columns 1 … F | features x |
| columns F+1 … 2F | negated features ¬x |
| further columns | unused: include 0, searched with 1 |
| row groups | class-major groups of an even number of rows: even rows vote for the class, odd rows against |

The valid column handles a difference between the two worlds. An empty clause (no
literal included) outputs 0 at inference in a Tsetlin machine, but an empty row
would always match in a CAM. Setting its valid bit makes it never match.

Capacity, with column 0 reserved:

| Macro | Literal columns | Boolean features | Rows |
|---|---|---|---|
| 64 × 80 (`muninn_tcam_dense_64x80`, `muninn_tcam_conservative_64x80`) | 80 | 39 | 64 |
| 2 × 64 × 80, tiled | 160 | 79 | 64 |

## Booleanization

Real-valued inputs become bits with a thermometer code. Bit k of a feature is 1 when
the value exceeds threshold k, and the thresholds are quantiles of the training
data. Values and thresholds are integers (`round(value × scale)`), so the firmware
and the Python reference set exactly the same bits.

Text (a console line) becomes hashed character bigrams (`muninn_ref.text`): after
lowercasing and collapsing whitespace, every pair of adjacent characters, with `^`
and `$` marking start and end, sets bit `(31·a + b) mod F`. A typo changes only two
or three pairs, so most bits of a command survive it. The exported header contains
the same function in C (`muninn_bigrams()`); a test compiles it and compares the bits.

Inputs that belong to no class need a class of their own ("unknown", trained on
counterexamples) and, optionally, a minimum winning sum. Both are exported as the
model's reject rule.

## Training a model

1. Choose the macro and check the feature budget: 2F + 1 ≤ literal columns.
2. Booleanize (`muninn_ref.booleanize.Thermometer`, or your own encoding).
3. Train `muninn_ref.tm.TsetlinMachine(classes, clauses, threshold, specificity)`.
   Clauses per class must be even and fit the row group (`Layout.rows_per_class`).
4. Export with `muninn_ref.hardware.export`. Check that `hardware.predict` agrees
   with `TsetlinMachine.predict` on all data. The reference examples treat any
   disagreement as a failure.
5. Write the firmware header with `muninn_ref.export.to_header`.

Hyperparameters to start with: threshold T ≈ clauses per class / 2, specificity
s = 3–4. More clauses per class help until the row group is full.
