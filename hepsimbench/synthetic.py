"""Synthetic example of the benchmark layout: toy histograms (Poisson draws from smooth shapes) and a toy shower
workflow; no experiment content. It is used by the tests and the Codabench dry run, and as a template for real
references, which are kept outside this repository.

    python -m hepsimbench.synthetic OUTDIR

writes OUTDIR/reference/ (two toy workflows WA and WB, each with a stock arm and a reseeded floor, and a toy shower
workflow WC with per-object features, a peak fit and a power rule) and OUTDIR/submissions/{good,shifted,slow}/ .
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

WORKFLOWS = {"WA": {"stock_rate_per_slot": 12.5, "mode": "same_events", "boundary": "toy primaries -> toy hits"},
             "WB": {"stock_rate_per_slot": 3.2, "mode": "shape", "boundary": "toy hits -> toy tracks"}}
HISTS = {"energy": (40, 0.0, 10.0, lambda c: np.exp(-c / 2.5)),
         "time": (50, 0.0, 1000.0, lambda c: 1 + 0.5 * np.sin(c / 160.0)),
         "multiplicity": (20, -0.5, 19.5, lambda c: np.exp(-0.5 * ((c - 6) / 2.5) ** 2))}
N_SHARDS, EVENTS_PER_SHARD = 8, 4000

# toy shower workflow: a leading cluster with a peaked energy in most events, plus soft extra clusters
PEAK = {"workflow": "WC", "stock_rate_per_slot": 55.8, "events": 60000, "block": 1500, "feature_events": 5000,
        "e_min": 10.0, "tolerance": 0.002}
PEAK_REFERENCE = {
    "task": PEAK["workflow"],
    "boundary": "toy particle at a calorimeter face -> toy clusters",
    "rate_unit": "events/s",
    "mode": "shape",
    "min_count": 10,
    "yield_tolerance": PEAK["tolerance"],
    "yield_max_se": {"peak": PEAK["tolerance"] / 3.0},
    "peak_fit": {"peak": {"start_above": 75.0, "start_width": 3.0}},
    "classifier": {"file": "features.npz", "seed": 20261118},
    "max_memory_mb_per_thread": 250,
}
PEAK_EDGES = {"cluster_energy": np.arange(0.0, 117.0, 2.0), "cluster_radius": np.arange(300.0, 701.0, 20.0),
              "cluster_size": np.arange(-0.5, 25.5, 1.0), "clusters_per_event": np.arange(-0.5, 6.5, 1.0)}
PEAK_FIT_EDGES = np.arange(40.0, 116.0, 1.0)
FEATURES = ("energy", "x", "y", "size", "seed_frac")


def _shard_hist(shard, values, edges, n_shards):
    return np.histogram2d(shard, values, bins=[np.arange(n_shards + 1) - 0.5, edges])[0]


def _shards(rows) -> dict:
    return {f"b{i:05d}": [float(v) for v in row] for i, row in enumerate(rows)}


def _peak_arm(seed: int, shift: float = 0.0, scale: float = 1.0) -> tuple[dict, dict]:
    """Toy shower arm: shift moves the shape features, scale the energies. Features are kept for the first
    feature_events events."""
    rng = np.random.default_rng(seed)
    n, block, e_min = PEAK["events"], PEAK["block"], PEAK["e_min"]
    lead = rng.random(n) < 0.93
    n_extra = rng.poisson(0.5, n)
    ev = np.r_[np.flatnonzero(lead), np.repeat(np.arange(n), n_extra)]
    m = ev.size
    is_lead = np.arange(m) < lead.sum()
    energy = np.where(is_lead, 88.0 + 2.2 * rng.standard_normal(m) - (rng.random(m) < 0.2) * rng.exponential(6.0, m),
                      1.0 + rng.exponential(5.0, m)) * scale
    energy = np.clip(energy, 0.5, None)
    r, phi = rng.uniform(380.0, 640.0, m), rng.uniform(0, 2 * np.pi, m)
    size = 1 + rng.poisson(1.0 + energy / (25.0 - 5.0 * shift), m)
    frac = np.clip(rng.beta(6.0 + 4.0 * shift, 4.0, m), 1.0 / size, 1.0)
    o = np.argsort(ev, kind="stable")
    ev, energy, r, phi, size, frac = ev[o], energy[o], r[o], phi[o], size[o], frac[o]
    n_shards = (n + block - 1) // block
    sh = ev // block
    sel = energy > e_min
    hists = {"cluster_energy": _shard_hist(sh, energy, PEAK_EDGES["cluster_energy"], n_shards),
             "cluster_radius": _shard_hist(sh[sel], r[sel], PEAK_EDGES["cluster_radius"], n_shards),
             "cluster_size": _shard_hist(sh[sel], size[sel], PEAK_EDGES["cluster_size"], n_shards),
             "clusters_per_event": _shard_hist(np.arange(n) // block, np.bincount(ev[sel], minlength=n),
                                               PEAK_EDGES["clusters_per_event"], n_shards)}
    best = np.zeros(n)
    np.maximum.at(best, ev, energy)
    has = best > e_min
    out = {"hists": {k: {"edges": PEAK_EDGES[k].tolist(), "shards": _shards(v)} for k, v in hists.items()},
           "yields": {"events_cluster_gt10": {"shards": {f"b{i:05d}": int(v) for i, v in enumerate(
               np.bincount(np.arange(n)[has] // block, minlength=n_shards))}}},
           "peaks": {"peak": {"edges": PEAK_FIT_EDGES.tolist(), "shards": _shards(
               _shard_hist(np.arange(n)[has] // block, best[has], PEAK_FIT_EDGES, n_shards))}}}
    x = np.column_stack([energy, r * np.cos(phi), r * np.sin(phi), size, frac])[sel]
    keep = ev[sel] < PEAK["feature_events"]
    return out, {"X": x[keep].astype(np.float32), "names": np.array(FEATURES), "event": ev[sel][keep]}


def _write_peak(d: Path, arm: tuple[dict, dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "hists.json").write_text(json.dumps(arm[0]))
    np.savez_compressed(d / "features.npz", **arm[1])


def _arm(seed: int, shift: float = 0.0, yield_scale: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    hists = {}
    for key, (nb, lo, hi, f) in HISTS.items():
        edges = np.linspace(lo, hi, nb + 1)
        c = 0.5 * (edges[1:] + edges[:-1])
        pdf = f(c + shift * (hi - lo) / nb)
        pdf = pdf / pdf.sum()
        shards = {f"s{i:02d}": rng.poisson(EVENTS_PER_SHARD * pdf).tolist() for i in range(N_SHARDS)}
        hists[key] = {"edges": edges.tolist(), "shards": shards}
    yields = {"selected": {"shards": {f"s{i:02d}": int(rng.poisson(900 * yield_scale)) for i in range(N_SHARDS)}}}
    return {"hists": hists, "yields": yields}


def _bench(w: str, dtype: str, rate: float, device: str) -> dict:
    return {"workflow": w, "boundary": WORKFLOWS[w]["boundary"], "warm_rate": rate, "dtype": dtype,
            "device": device, "inputs": "fresh"}


def write(out: Path) -> Path:
    out = Path(out)
    ref = out / "reference"
    for i, (w, cfg) in enumerate(WORKFLOWS.items()):
        (ref / w / "stock").mkdir(parents=True, exist_ok=True)
        (ref / w / "floor").mkdir(parents=True, exist_ok=True)
        (ref / w / "reference.json").write_text(json.dumps(
            {**cfg, "yield_tolerance": 0.02, "min_count": 5, "su_per_hour": {"toy-gpu": 3.0, "toy-cpu x8": 1.0}}))
        (ref / w / "stock" / "hists.json").write_text(json.dumps(_arm(100 + i)))
        (ref / w / "floor" / "hists.json").write_text(json.dumps(_arm(200 + i)))
    w = PEAK["workflow"]
    (ref / w).mkdir(parents=True, exist_ok=True)
    (ref / w / "reference.json").write_text(json.dumps(
        {**PEAK_REFERENCE, "stock_rate_per_slot": PEAK["stock_rate_per_slot"], "su_per_hour": {"toy-gpu": 3.0}}))
    _write_peak(ref / w / "stock", _peak_arm(500))
    _write_peak(ref / w / "floor", _peak_arm(600))
    subs = {"good": dict(shift=0.0, ys=1.0, rate=5000.0), "shifted": dict(shift=0.6, ys=0.9, rate=5000.0),
            "slow": dict(shift=0.0, ys=1.0, rate=40.0)}
    for name, s in subs.items():
        for i, wf in enumerate(WORKFLOWS):
            for j, dt in enumerate(("float32", "float64")):
                d = out / "submissions" / name / wf / dt
                d.mkdir(parents=True, exist_ok=True)
                (d / "hists.json").write_text(json.dumps(_arm(300 + 10 * i + j, s["shift"], s["ys"])))
                (d / "bench.json").write_text(json.dumps(_bench(wf, dt, s["rate"] * (1.6 if dt == "float32" else 1),
                                                                "toy-gpu")))
        for j, dt in enumerate(("float32", "float64")):
            d = out / "submissions" / name / w / dt
            _write_peak(d, _peak_arm(700 + j, s["shift"], 0.99 if s["shift"] else 1.0))
            (d / "bench.json").write_text(json.dumps(
                {"workflow": w, "boundary": PEAK_REFERENCE["boundary"], "dtype": dt, "device": "toy-gpu",
                 "warm_rate": s["rate"] * (1.6 if dt == "float32" else 1), "inputs": "fresh", "host": "device"}))
    return out


if __name__ == "__main__":
    print(write(Path(sys.argv[1])))
