# SPDX-FileCopyrightText: 2026 aesc silicon
#
# SPDX-License-Identifier: Apache-2.0

from muninn_ref import export
from muninn_ref.__main__ import main
from tsetlin.iris import train


def test_json_round_trip(tmp_path):
    train.run(seed=1, out=tmp_path)
    m = export.load(tmp_path / "iris.json")
    assert m.name == "iris" and m.layout.rows == 64 and m.layout.rows_per_class == 20
    X_train, X_test, y_train, y_test = train.dataset(1)
    acc = (m.predict(X_test) == y_test).mean()
    assert abs(acc - m.meta["test_accuracy"]) < 1e-9
    # regenerated header is identical to the one written at training
    export.to_header(tmp_path / "again.h", m.name, m.layout, m.rows, m.booleanizer, m.labels)
    assert (tmp_path / "again.h").read_text() == (tmp_path / "iris.h").read_text()


def test_cli(tmp_path, capsys):
    assert main(["train", "iris", "--epochs", "60", "-o", str(tmp_path)]) == 0
    js = str(tmp_path / "iris.json")
    assert main(["check", js]) == 0
    assert main(["info", js]) == 0
    assert main(["classify", js, "5.1", "3.5", "1.4", "0.2"]) == 0   # setosa
    assert capsys.readouterr().out.splitlines()[-1].startswith("class 0 (setosa)")
    assert main(["header", js, "-o", str(tmp_path / "x.h")]) == 0
    assert (tmp_path / "x.h").read_text() == (tmp_path / "iris.h").read_text()


def test_console(tmp_path, capsys, monkeypatch):
    import io
    train.run(seed=1, out=tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("5.1 3.5 1.4 0.2\n1 2\n\n6.3,3.3,6.0,2.5\nq\n"))
    assert main(["console", str(tmp_path / "iris.json")]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("class 0 (setosa)")
    assert out[1].startswith("error: expected 4 values")
    assert out[2].startswith("class 2 (virginica)")


def test_load_rejects_wide_rows(tmp_path):
    train.run(seed=1, out=tmp_path)
    m = export.load(tmp_path / "iris.json")
    rows = list(m.rows)
    rows[0] |= 1 << m.layout.literals
    export.to_json(tmp_path / "bad.json", "iris", m.layout, rows, m.booleanizer)
    try:
        export.load(tmp_path / "bad.json")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
