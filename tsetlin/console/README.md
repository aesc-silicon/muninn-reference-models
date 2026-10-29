<!--
SPDX-FileCopyrightText: 2026 aesc silicon

SPDX-License-Identifier: CC-BY-4.0
-->

# Console commands

Typo-tolerant commands for the ElemRV bare-metal shell: `help`, `version`, `led on`,
`led off`, `clear`. Anything else is "unknown command". The ElemRV-C demo runs
`console_64x80` on its `muninn_tcam_dense_64x80`.

| | `console_64x80` | `console_64x160` |
|---|---|---|
| Macro | 64 × 80 | 2 × 64 × 80, tiled |
| Booleanization | 39 hashed bigrams, 79 literals | 79 hashed bigrams, 159 literals |
| Clauses | 6 classes × 10 (rows 60–63 unused) | same |
| Training | T = 6, s = 1.5, 100 epochs, 3000/2000 samples, seed 1 | same |
| Commands recognized (test, ≤ 2 typos) | 95.6 % | 96.3 % |
| Unknown input rejected | 82.5 % | 92.2 % |
| Exact commands | 5 of 5 | 5 of 5 |
| Hardware reference | 0 disagreements | 0 disagreements |

**Data.** Every sample is a command with 0–2 typos (neighbouring key, dropped,
doubled, swapped or extra key) or an unknown input: another shell word (`ls`,
`reboot`, `status`, …), two of them, or random letters, always more than two edits
away from every command.

**Reject rule.** An input is "unknown" when the unknown class wins or the winning
sum is below 1. Ties go to the lowest class index, as in `TsetlinCtrl`. A low
specificity (s = 1.5) keeps clauses short, which is what makes them survive typos.

**Firmware.** `model/console_*.h` holds the include bits, the labels, the reject rule
(`*_UNKNOWN`, `*_MIN_SUM`) and `muninn_bigrams()`, which turns a line into the
feature bits; the search vector is built from them like for every model (valid
column 0, features, negated features, unused columns 1).

```bash
python3 -m muninn_ref train console                     # both macros, ~40 s
python3 -m muninn_ref console tsetlin/console/model/console_64x160.json
> hlep
class 0 (help)  sums +4 -5 -5 -5 -5 -1
> lde off
class 3 (led off)  sums -5 -4 -3 +5 -5 +0
> reboot
unknown  sums -4 -2 -4 -2 -3 +3
```
