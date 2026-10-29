# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

"""Text input: hashed character bigrams, and synthetic typos to train against.

A command line is lowercased (ASCII), runs of whitespace become one space, leading and
trailing whitespace is dropped. With '^' before and '$' after the line, every pair of
adjacent characters (a, b) sets feature (31 * a + b) % features. Typos change only a few
pairs, so most features of a command survive them.

The C version in the exported header (muninn_bigrams) computes the same bits.
"""

import numpy as np

SPACE = b" \t\r\n"
MUL = 31


class Bigrams:
    def __init__(self, features):
        self.features = features

    def pairs(self, line):
        """Character pairs of the normalized line, exactly as the C version walks it."""
        out, prev, space = [], ord("^"), False
        for c in line.encode("ascii", "replace"):
            if c in SPACE:
                space = prev != ord("^")
                continue
            if ord("A") <= c <= ord("Z"):
                c += ord("a") - ord("A")
            if space:
                out.append((prev, ord(" ")))
                prev, space = ord(" "), False
            out.append((prev, c))
            prev = c
        out.append((prev, ord("$")))
        return out

    def bits(self, line):
        v = np.zeros(self.features, dtype=bool)
        for a, b in self.pairs(line):
            v[(MUL * a + b) % self.features] = True
        return v

    def transform(self, lines):
        return np.array([self.bits(s) for s in lines]).reshape(len(lines), self.features)


# ---- typos ------------------------------------------------------------------------
KEYBOARD = ["1234567890-", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
_POS = {c: (r, i) for r, row in enumerate(KEYBOARD) for i, c in enumerate(row)}
NEIGHBOURS = {c: [d for d in _POS if d != c and abs(_POS[d][0] - _POS[c][0]) <= 1
                  and abs(_POS[d][1] - _POS[c][1]) <= 1] for c in _POS}


def typo(word, rng):
    """One typing error: neighbouring key, dropped, doubled, swapped or extra key."""
    i = int(rng.integers(len(word)))
    c, op = word[i], int(rng.integers(5))
    if op == 0 and c in NEIGHBOURS:
        return word[:i] + rng.choice(NEIGHBOURS[c]) + word[i + 1:]
    if op == 1 and len(word) > 2:
        return word[:i] + word[i + 1:]
    if op == 2:
        return word[:i] + c + word[i:]
    if op == 3 and i + 1 < len(word):
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if c in NEIGHBOURS:
        return word[:i] + rng.choice(NEIGHBOURS[c]) + c + word[i:]
    return word


def distance(a, b):
    """Levenshtein distance."""
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]
