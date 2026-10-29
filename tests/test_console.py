# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

import io
import shutil
import subprocess

import numpy as np
import pytest

from muninn_ref import export
from muninn_ref.__main__ import main
from muninn_ref.text import Bigrams, distance, typo
from tsetlin.console import train

LINES = ["help", "  LED   on ", "led\toff\r\n", "", "x", "vresion", "ümlaut ok"]


def test_bigrams():
    b = Bigrams(79)
    assert b.pairs("Led  On") == b.pairs(" led on\n")
    assert [(chr(a), chr(c)) for a, c in b.pairs("a b")] == [("^", "a"), ("a", " "),
                                                            (" ", "b"), ("b", "$")]
    assert b.transform(LINES).shape == (len(LINES), 79)


def test_typos():
    rng = np.random.default_rng(0)
    words = [typo("version", rng) for _ in range(200)]
    assert all(distance(w, "version") <= 2 for w in words)       # swap counts as 2
    assert len(set(words)) > 30
    assert distance("led on", "led off") == 2


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    out = tmp_path_factory.mktemp("console")
    res = train.run((64, 160), seed=1, epochs=30, out=out, samples=(1500, 500))
    return out, res


def test_console_model(model):
    out, res = model
    assert res["hardware_disagreements"] == 0
    assert res["exact_commands"] == len(train.COMMANDS)
    assert res["commands_recognized"] >= 0.9
    assert res["unknown_rejected"] >= 0.75
    m = export.load(out / "console_64x160.json")
    assert m.text and m.labels[-1] == "unknown" and m.reject["class"] == 5
    assert main(["check", str(out / "console_64x160.json")]) == 0


def test_console_cli(model, capsys, monkeypatch):
    out, _ = model
    monkeypatch.setattr("sys.stdin", io.StringIO("led on\nhepl\nreboot\n"))
    assert main(["console", str(out / "console_64x160.json")]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("class 2 (led on)")
    assert lines[1].startswith("class 0 (help)")
    assert lines[2].startswith("unknown")
    assert main(["classify", str(out / "console_64x160.json"), "led", "off"]) == 0
    assert capsys.readouterr().out.startswith("class 3 (led off)")


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not found")
def test_c_bigrams_match(model, tmp_path):
    """The header's muninn_bigrams() sets the same bits as muninn_ref.text."""
    out, _ = model
    shutil.copy(out / "console_64x160.h", tmp_path)
    cases = ", ".join('"' + s.encode("unicode_escape").decode().replace('"', '\\"') + '"'
                      for s in LINES)
    (tmp_path / "t.c").write_text(f"""#include <stdio.h>
#include "console_64x160.h"
int main(void)
{{
	static const char *lines[] = {{ {cases} }};
	unsigned int bits[3];

	for (unsigned int i = 0; i < sizeof(lines) / sizeof(lines[0]); i++) {{
		muninn_bigrams(lines[i], CONSOLE_64X160_FEATURES, bits);
		printf("%08x %08x %08x\\n", bits[0], bits[1], bits[2]);
	}}
	return console_64x160_labels[CONSOLE_64X160_UNKNOWN][0] != 'u';
}}
""")
    subprocess.run(["gcc", "-std=c99", "-Wall", "-Werror", "-I", str(tmp_path),
                    str(tmp_path / "t.c"), "-o", str(tmp_path / "t")], check=True)
    run = subprocess.run([str(tmp_path / "t")], capture_output=True, text=True)
    assert run.returncode == 0
    b = Bigrams(79)
    for line, got in zip(LINES, run.stdout.splitlines()):
        v = sum(1 << i for i, x in enumerate(b.bits(line)) if x)
        assert got == " ".join(f"{w:08x}" for w in export.words(v, 3)), repr(line)
