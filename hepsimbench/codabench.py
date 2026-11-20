"""Build a Codabench bundle (results submission, one phase) for a fast-simulation benchmark.

The reference data is the stock side; an experiment's real reference stays private. By default the bundle is built
on the synthetic example (hepsimbench.synthetic), so the bundle can be built and tested anywhere. With --reference DIR it holds a
real reference tree (kept outside any public repository), and the bundle is for an unpublished competition or local
scoring only. Leaderboard columns follow the reference's workflows. SU per event is never scored here.

    python -m hepsimbench.codabench OUTDIR [--reference DIR] [--title TEXT]

writes OUTDIR/bundle/ and OUTDIR/competition_bundle.zip (plus OUTDIR/example/ when the synthetic example is used).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TITLE = "Fast-simulation three metrics (PLACEHOLDER TITLE)"

SCORING = '''"""Codabench scoring program: the submission zip holds <W>/<dtype>/{hists.json,bench.json[,features.npz]}."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CODABENCH_ROOT", "/app"))
sys.path.insert(0, str(ROOT / "program"))
from score import ScoreError, leaderboard_row, score  # noqa: E402


def main():
    res_dir, ref_dir = ROOT / "input" / "res", ROOT / "input" / "ref"
    wfs = [p.name for p in ref_dir.iterdir() if (p / "reference.json").is_file()]
    kids = [p for p in res_dir.iterdir() if p.is_dir()]
    if not any((res_dir / w).is_dir() for w in wfs) and len(kids) == 1:
        res_dir = kids[0]                               # tolerate one wrapping folder
    try:
        res = score(res_dir, ref_dir, private=False)
    except ScoreError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    (ROOT / "output").mkdir(parents=True, exist_ok=True)
    (ROOT / "output" / "scores.json").write_text(json.dumps(leaderboard_row(res), indent=1))
    (ROOT / "output" / "detailed_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(leaderboard_row(res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def competition_yaml(workflows: list[str], title: str, cls_workflows=(), power_workflows=(),
                     commit_title: str = "commit of the fast simulation", task_name: str = "three metrics") -> str:
    """Leaderboard columns: the overall scores, then validity and multiplier per workflow and dtype, plus the
    classifier score (workflows whose reference.json has a classifier) and the power flag (workflows with a
    yield_max_se rule)."""
    cols = [("Validity score", "validity_score", "desc"), ("Min validity", "min_validity", "desc"),
            ("Throughput x (self-reported)", "throughput_geomean", "desc")]
    if cls_workflows:
        cols.append(("Classifier score", "cls_score", "desc"))
    for w in workflows:
        for dt in ("f32", "f64"):
            cols += [(f"{w} {dt} validity", f"{w.lower()}_{dt}_validity", "desc"),
                     (f"{w} {dt} x", f"{w.lower()}_{dt}_multiplier", "desc")]
            if w in cls_workflows:
                cols.append((f"{w} {dt} classifier", f"{w.lower()}_{dt}_cls_score", "desc"))
            if w in power_workflows:
                cols.append((f"{w} {dt} power (1 = se small enough)", f"{w.lower()}_{dt}_power", "desc"))
    col_lines = "\n".join(f"      - {{title: \"{t}\", key: {k}, index: {i}, sorting: {s}, precision: 3}}"
                          for i, (t, k, s) in enumerate(cols))
    return f"""version: 2
# PLACEHOLDERS before any use: logo.png, terms, dates. The reference data is private: unpublished competition only.
title: "{title}"
description: >-
  Validity (DQM-style histogram flag rule against stock with a reseeded-stock floor, plus yields with a
  statistical-power rule where set), the classifier score where the reference has per-cluster features, and the
  throughput multiplier against stock grid slots for a fast-simulation commit, both float32 and float64.
image: logo.png
terms: pages/terms.md
docker_image: codalab/codalab-legacy:py312
registration_auto_approve: false
enable_detailed_results: true
pages:
  - {{title: Overview, file: pages/overview.md}}
  - {{title: Terms, file: pages/terms.md}}
tasks:
  - {{index: 0, name: {task_name}, scoring_program: scoring_program, reference_data: reference_data}}
solutions:
  - {{index: 0, tasks: [0], path: solution}}
phases:
  - index: 0
    name: Commits
    start: 2027-01-01 00:00:00   # PLACEHOLDER
    tasks: [0]
    max_submissions_per_day: 20
    execution_time_limit: 300
    accepts_only_result_submissions: true
leaderboards:
  - title: Results
    key: main
    submission_rule: Force_Last
    columns:
{col_lines}
fact_sheet:
  commit:
    key: commit
    type: text
    title: {commit_title}
    selection: ""
    is_required: "true"
    is_on_leaderboard: "true"
"""


def zip_tree(src: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, arcname=f.relative_to(src).as_posix())


def build(out: Path, reference: Path | None = None, title: str = DEFAULT_TITLE, solution: Path | None = None, *,
          synthetic=None, overview: Path | None = None, commit_title: str = "commit of the fast simulation",
          task_name: str = "three metrics") -> Path:
    """The bundle; synthetic (a module with write(dir)) makes the example when no reference is given (default
    hepsimbench.synthetic), overview is the markdown of the Overview page (default the package README)."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if reference is None:
        if synthetic is None:
            from . import synthetic
        ex = synthetic.write(out / "example")
        reference, solution = ex / "reference", solution or ex / "submissions" / "good"
    if solution is None:
        raise SystemExit("--solution is required with --reference (a submission tree that scores on it)")
    b = out / "bundle"
    (b / "scoring_program").mkdir(parents=True)
    shutil.copy(HERE / "score.py", b / "scoring_program" / "score.py")
    (b / "scoring_program" / "scoring.py").write_text(SCORING)
    (b / "scoring_program" / "metadata.yaml").write_text("command: python3 scoring.py\n")
    shutil.copytree(reference, b / "reference_data")
    shutil.copytree(solution, b / "solution")
    workflows = sorted(p.name for p in reference.iterdir() if (p / "reference.json").is_file())
    cfgs = {w: json.loads((reference / w / "reference.json").read_text()) for w in workflows}
    cls_w = [w for w in workflows if cfgs[w].get("classifier")]
    power_w = [w for w in workflows if cfgs[w].get("yield_max_se")]
    (b / "competition.yaml").write_text(competition_yaml(workflows, title, cls_w, power_w, commit_title, task_name))
    (b / "pages").mkdir()
    overview = overview or HERE / "README_codabench.md"
    (b / "pages" / "overview.md").write_text(Path(overview).read_text())
    (b / "pages" / "terms.md").write_text("# Terms (PLACEHOLDER)\n\nNot real terms; to be replaced before any use.\n")
    zip_tree(b, out / "competition_bundle.zip")
    return out / "competition_bundle.zip"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--solution", type=Path)
    ap.add_argument("--title", default=DEFAULT_TITLE)
    a = ap.parse_args(argv)
    print(build(a.out, a.reference, a.title, a.solution))
    return 0


if __name__ == "__main__":
    sys.exit(main())
