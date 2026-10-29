# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess

import pytest

from muninn_ref import export
from tsetlin.iris import train


def test_iris_reference(tmp_path):
    res = train.run(seed=1, out=tmp_path)
    assert res["test_accuracy"] >= 0.9
    assert res["hardware_disagreements"] == 0
    assert res["literals_used"] <= 79                       # fits a 64 x 80 macro
    assert (tmp_path / "iris.json").is_file()


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not found")
def test_header_compiles(tmp_path):
    train.run(seed=1, out=tmp_path)
    (tmp_path / "t.c").write_text('#include "iris.h"\nint main(void) { '
                                  'return iris_include[0][0] == 0 || IRIS_WORDS != 3; }\n')
    subprocess.run(["gcc", "-Wall", "-Werror", "-I", str(tmp_path), str(tmp_path / "t.c"),
                    "-o", str(tmp_path / "t")], check=True)
    assert subprocess.run([str(tmp_path / "t")]).returncode == 0


def test_words():
    assert export.words((1 << 40) | 5, 3) == [5, 1 << 8, 0]
