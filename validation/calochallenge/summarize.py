"""Score every arm against the CaloChallenge reference and write the run record and a Markdown table.

    python validation/calochallenge/summarize.py OUT_DIR [--criteria validation/calochallenge/criteria.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hepsimbench.score import score

WORKFLOW = "CCD1P"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--criteria", type=Path, default=Path(__file__).with_name("criteria.json"))
    ap.add_argument("--require-checks", action="store_true", help="exit 1 unless the null and dtype checks pass")
    a = ap.parse_args(argv)
    crit_bytes = a.criteria.read_bytes()
    crit = json.loads(crit_bytes)
    ref = a.out / "reference"
    rows, record = [], {"criteria_sha256": hashlib.sha256(crit_bytes).hexdigest(), "arms": {}}
    for sub in sorted((a.out / "submissions").iterdir()):
        res = score(sub, ref)
        legs = {}
        for dt in ("float32", "float64"):
            r = res["workflows"][WORKFLOW][dt]
            legs[dt] = {"validity": r["validity"], "n_hists": r["n_hists"], "n_flagged": r["n_flagged"],
                        "cls_score": r.get("classifier", {}).get("cls_score"),
                        "auc": r.get("classifier", {}).get("auc"),
                        "floor_cls_score": r.get("classifier", {}).get("floor_cls_score")}
        record["arms"][sub.name] = legs
        rows.append((sub.name, legs))
    null = record["arms"].get(crit["null_arm"])
    checks = {}
    if null:
        checks["null"] = all(l["validity"] >= crit["null_must"]["validity_min"] and
                             (l["cls_score"] or 0) >= crit["null_must"]["cls_score_min"] for l in null.values())
    checks["dtype_legs"] = all(abs(l["float32"]["validity"] - l["float64"]["validity"]) <= 0.02
                               for l in record["arms"].values())
    record["checks"] = checks
    (a.out / "record.json").write_text(json.dumps(record, indent=1))
    lines = ["| Arm | validity f32 | validity f64 | flagged / hists (f64) | cls_score (f64) | AUC (f64) |",
             "|---|---|---|---|---|---|"]
    for name, legs in sorted(rows, key=lambda r: -(r[1]["float64"]["cls_score"] or 0)):
        f, d = legs["float32"], legs["float64"]
        lines.append(f"| {name} | {f['validity']:.3f} | {d['validity']:.3f} | {d['n_flagged']} / {d['n_hists']} | "
                     f"{d['cls_score']:.3f} | {d['auc']:.3f} |")
    (a.out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("checks:", checks)
    return 1 if a.require_checks and not all(checks.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
