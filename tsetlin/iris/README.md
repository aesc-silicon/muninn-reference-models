<!--
SPDX-FileCopyrightText: 2026 aesc silicon

SPDX-License-Identifier: CC-BY-4.0
-->

# Iris

The "hello world" of the reference chain. Fisher's Iris data (bundled with
scikit-learn) has 150 flowers, 4 measurements each, and 3 species.

| | |
|---|---|
| Macro | 64 × 80 (`muninn_tcam_dense_64x80`, `muninn_tcam_conservative_64x80`) |
| Booleanization | 4-bit thermometer per measurement (scale 10: exact integers) → 16 features, 33 literals |
| Clauses | 3 classes × 20 (rows 60–63 unused) |
| Training | T = 10, s = 3.0, 60 epochs, 70/30 split, seed 1 |
| Accuracy (seed 1) | train 94.3 %, test 97.8 % |
| Hardware reference | 0 disagreements with the Tsetlin machine on all 150 samples |
| Include bits | 491 of 64 × 80 |

Over 10 seeds the test accuracy averages 92.4 % (`compare_tmu.py`). `tmu` 0.7.1 with
the same data and settings reaches 89.1 %.

```bash
python3 -m muninn_ref train iris          # writes model/iris.json and model/iris.h
python3 -m muninn_ref check tsetlin/iris/model/iris.json   # re-check without retraining
python3 tsetlin/iris/compare_tmu.py      # needs tmu (reference container)
```

The exported model needs one change in the Nafarr frontend: 3 classes do not divide
64 rows, so `TsetlinCtrl` must accept a group size per class with unused rows
behind it, as `muninn_ref.hardware.Layout` does.
