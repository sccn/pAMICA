"""Run the reference binary on both forms of a pinned ``input.param``
(issue #351, after epic #324 Phase 17).

Phase 17 (#354) rebuilt ``AMICANative``'s parameter table, so a pinned call
now writes its keys in a new order and adds ``do_approx_sphere 1``, the value
compiled into the binary (``amica15_header.f90``), with every other key and
value unchanged (``pin_check.py --against``). The binary reads keys by name,
so the two files should give the same run. This script checks that on the
bundled sample: for each pinned configuration it lets ``AMICANative`` write
its current ``input.param``, runs the binary on it, then runs it again in a
copy of the work directory whose ``input.param`` has the pre-Phase-17 key
order and no ``do_approx_sphere`` line, and compares every output file byte
for byte, except that the per-iteration wall-clock times are removed from
``out.txt`` first. Seeded and single-threaded, so each run is deterministic.

    uv run python .context/issue-351/pin_run_check.py OUT.json
"""

from __future__ import annotations

import filecmp
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SAVED = HERE / "raw/pin_check_pinned_params_pre_phase17.json"
# The end of each iteration line of out.txt, e.g. "(  0.19 s,   0.0 h)".
TIMING = re.compile(r"\(\s*[0-9.]+ s,\s*[0-9.]+ h\)")


def old_form(new_text: str, old_text: str) -> str:
    """``new_text``'s lines in ``old_text``'s key order, without the key
    Phase 17 added; keys ``old_text`` lacks (``seed``) keep their place last."""
    lines = {ln.split(None, 1)[0]: ln for ln in new_text.splitlines() if ln.strip()}
    lines.pop("do_approx_sphere")
    order = [ln.split(None, 1)[0] for ln in old_text.splitlines() if ln.strip()]
    keys = [k for k in order if k in lines] + [k for k in lines if k not in order]
    return "\n".join(lines[k] for k in keys) + "\n"


def compare_run(name: str, kwargs: dict, data, saved: dict) -> dict:
    import pamica.native.engine as engine

    real = engine.subprocess.run
    record: dict = {}

    def both_forms(cmd, cwd=None, **kw):
        work = Path(cwd)
        twin = work.parent / (work.name + "_old_form")
        shutil.copytree(work, twin)
        new_text = (work / "input.param").read_text()
        (twin / "input.param").write_text(old_form(new_text, saved[name]))
        record["new_has_approx"] = "do_approx_sphere 1" in new_text
        proc_old = real(cmd, cwd=twin, **kw)
        proc = real(cmd, cwd=cwd, **kw)
        a, b = work / "amicaout", twin / "amicaout"
        files = sorted(p.name for p in a.iterdir())
        binary = [f for f in files if f != "out.txt"]
        match, mismatch, errors = filecmp.cmpfiles(a, b, binary, shallow=False)
        logs = [TIMING.sub("", (d / "out.txt").read_text()) for d in (a, b)]
        (match if logs[0] == logs[1] else mismatch).append("out.txt (times removed)")
        record.update(
            files=files, identical=match, different=mismatch, missing=errors,
            returncodes=[proc.returncode, proc_old.returncode],
        )  # fmt: skip
        shutil.rmtree(twin)
        return proc

    engine.subprocess.run = both_forms
    try:
        engine.AMICANative(**kwargs).fit(data)
    finally:
        engine.subprocess.run = real
    record["all_identical"] = (
        not record["different"] and not record["missing"] and bool(record["identical"])
    )
    return record


def main() -> None:
    import _reference_settings as rs
    import keep_best_ensemble as kb
    import newton_seeds as ns

    data = kb.load_data()
    saved = json.loads(SAVED.read_text())
    t1 = rs._table1()
    one = dict(threads=1, max_threads=1, timeout=3600, seed=1)
    configs = {
        "reproduce_table1 single-model sweep": {
            **one, **t1.single_model_reference_kwargs(20)},
        "reproduce_table1 multi-model ensemble": {
            **one, **t1.multimodel_reference_kwargs(20)},
        "newton_seeds reference": {**one, **ns.fortran_kwargs(60)},
    }  # fmt: skip
    out = {name: compare_run(name, kw, data, saved) for name, kw in configs.items()}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
    print(
        json.dumps(
            {k: (v["all_identical"], len(v["identical"])) for k, v in out.items()}
        )
    )
    if not all(v["all_identical"] for v in out.values()):
        raise SystemExit("the two forms of a pinned input.param gave different runs")


if __name__ == "__main__":
    main()
