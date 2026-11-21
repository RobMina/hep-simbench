"""CaloChallenge 2022 Dataset 1 (photons) in the hep-simbench layout: Geant4 as stock, the public submitted samples
as fast arms. Public inputs only (Zenodo 10.5281/zenodo.8099322 and 10.5281/zenodo.15961728, CC-BY-4.0).

    python validation/calochallenge/build.py DATA_DIR OUT_DIR [--criteria validation/calochallenge/criteria.json]

DATA_DIR holds dataset_1_photons_{1,2}.hdf5, the ds1-photons_<id>.hdf5 submissions and binning_dataset_1_photons.xml
(fetch.sh). OUT_DIR receives reference/CCD1P/{stock,floor}/ and submissions/<model>/CCD1P/{float32,float64}/, each
with hists.json and features.npz; score with `python -m hepsimbench.score OUT_DIR/submissions/<model> OUT_DIR/reference`.

Arms: stock = photons_2 (the evaluation file); floor = the first half of photons_1; the second half of photons_1 is
the null submission `geant4_null` (Geant4 against Geant4: it must pass). The histogram edges come from the floor only,
so they are fixed before any fast arm or the stock file is read. Features are computed from the voxel energies in
float32 and in float64, giving the two dtype legs (the submitted samples themselves are float32 files).

Per shower and per layer l (the five layers of the binning file): f_l = E_l / E_inc, the energy-weighted mean and rms
of the voxel radius r, the energy-weighted mean of x = r cos(alpha) and y = r sin(alpha) for layers with more than one
alpha bin, and the fraction of the layer's voxels with energy. Plus E_tot / E_inc. Histograms per incident-energy
class (the 15 energies) and over all energies; a layer with no energy enters only its energy and sparsity histograms.
Voxel order within a layer is alpha-major (alpha bin outer, r bin inner); the script checks that order against the
mean radial profile and refuses the inputs if the profile is not falling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

WORKFLOW = "CCD1P"
SUBMISSIONS = {1: "CaloDiffusion", 4: "CaloINN", 7: "Calo-VQ", 8: "CaloScore", 9: "CaloScore_distilled",
               10: "CaloScore_single-shot", 11: "CaloFlow_teacher", 12: "CaloFlow_student", 16: "CaloMan",
               17: "BoloGAN", 25: "CaloShower2GAN", 26: "CaloShower3GAN", 27: "CaloVAE+INN", 28: "CaloForest",
               29: "CaloGraph"}


def read_binning(path: Path) -> list[dict]:
    """Layers of the photon entry: id, r edges [mm], number of alpha bins, voxel slice."""
    root = ET.parse(path).getroot()
    part = next(p for p in root.iter("Particle") if p.get("name", "").lower().startswith("photon"))
    layers, start = [], 0
    for el in part.iter("Layer"):
        r = np.array([float(v) for v in el.get("r_edges").split(",")])
        na = int(el.get("n_bin_alpha"))
        n = (len(r) - 1) * na
        if n == 0:
            continue
        layers.append({"id": int(el.get("id")), "r_edges": r, "n_alpha": na, "sl": slice(start, start + n)})
        start += n
    return layers


def read_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py
    with h5py.File(path, "r") as f:
        return np.asarray(f["incident_energies"][:]).reshape(-1), np.asarray(f["showers"][:])


def voxel_geometry(layer: dict, dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """r, x, y of every voxel of a layer (alpha-major order)."""
    re = layer["r_edges"].astype(dtype)
    rc = 0.5 * (re[1:] + re[:-1])
    na = layer["n_alpha"]
    ac = (np.arange(na, dtype=dtype) + dtype(0.5)) * dtype(2 * np.pi / na)
    r = np.tile(rc, na)
    a = np.repeat(ac, len(rc))
    return r, r * np.cos(a), r * np.sin(a)


def check_order(layers: list[dict], showers: np.ndarray) -> None:
    """Refuse if the mean radial profile of the widest layer does not fall from the first to the last r bin."""
    lay = max(layers, key=lambda l: l["sl"].stop - l["sl"].start)
    nr = len(lay["r_edges"]) - 1
    e = showers[:, lay["sl"]].astype(np.float64).mean(axis=0).reshape(lay["n_alpha"], nr).mean(axis=0)
    area = np.diff(lay["r_edges"] ** 2)
    dens = e / area
    if not (dens[0] > dens[-1] and np.corrcoef(np.arange(nr), np.log(dens + 1e-30))[0, 1] < -0.5):
        raise SystemExit(f"layer {lay['id']}: radial energy density does not fall with r under the alpha-major "
                         f"voxel order; refusing the inputs")


def features(einc: np.ndarray, showers: np.ndarray, layers: list[dict], dtype) -> dict[str, np.ndarray]:
    s = showers.astype(dtype)
    einc = einc.astype(dtype)
    out = {"etot_over_einc": s.sum(axis=1) / einc}
    for lay in layers:
        v = s[:, lay["sl"]]
        el = v.sum(axis=1)
        r, x, y = voxel_geometry(lay, dtype)
        has = el > 0
        w = np.where(has[:, None], v / np.where(has, el, 1)[:, None], 0)
        tag = f"L{lay['id']}"
        out[f"{tag}_frac"] = el / einc
        out[f"{tag}_sparsity"] = (v > 0).mean(axis=1).astype(dtype)
        mr = (w * r).sum(axis=1)
        out[f"{tag}_r_mean"] = np.where(has, mr, np.nan)
        out[f"{tag}_r_rms"] = np.where(has, np.sqrt(np.clip((w * r * r).sum(axis=1) - mr * mr, 0, None)), np.nan)
        if lay["n_alpha"] > 1:
            out[f"{tag}_x_mean"] = np.where(has, (w * x).sum(axis=1), np.nan)
            out[f"{tag}_y_mean"] = np.where(has, (w * y).sum(axis=1), np.nan)
    return out


def energy_classes(einc: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Index of the nearest class energy (log scale); refuse showers more than 1 % away from every class."""
    d = np.abs(np.log(einc[:, None]) - np.log(energies[None, :]))
    k = d.argmin(axis=1)
    if np.any(d[np.arange(len(einc)), k] > 0.01):
        raise ValueError("incident energies outside the 15 classes of photons_1")
    return k


def make_edges(feat: dict, cls: np.ndarray, n_cls: int, nbins: int) -> dict:
    """Equal-probability edges per (feature, class) and inclusive, from the floor arm only."""
    edges = {}
    for name, v in feat.items():
        for c in list(range(n_cls)) + [None]:
            x = v if c is None else v[cls == c]
            x = x[np.isfinite(x)].astype(np.float64)
            if x.size < 10 * nbins:
                continue
            e = np.unique(np.quantile(x, np.linspace(0.001, 0.999, nbins + 1)))
            if e.size < 4:
                continue
            edges[f"{name}@{'all' if c is None else f'E{c:02d}'}"] = e
    return edges


def hists(feat: dict, cls: np.ndarray, edges: dict) -> dict:
    out = {}
    for key, e in edges.items():
        name, sel = key.split("@")
        v = feat[name].astype(np.float64)
        if sel != "all":
            v = v[cls == int(sel[1:])]
        v = v[np.isfinite(v)]
        v = np.clip(v, e[0], np.nextafter(e[-1], -np.inf))  # under/overflow into the end bins
        out[key] = {"edges": e.tolist(), "counts": np.histogram(v, e)[0].astype(int).tolist()}
    return out


def feature_matrix(feat: dict, einc: np.ndarray, names: list[str]) -> np.ndarray:
    cols = [np.log(einc.astype(np.float64))]
    for n in names:
        v = feat[n].astype(np.float64)
        cols.append(np.where(np.isfinite(v), v, 0.0))
    return np.column_stack(cols)


def write_arm(d: Path, feat, einc, cls, edges, names, max_rows, seed) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "hists.json").write_text(json.dumps({"hists": hists(feat, cls, edges)}))
    X = feature_matrix(feat, einc, names)
    rng = np.random.default_rng(seed)
    keep = rng.permutation(len(X))[:max_rows]
    np.savez_compressed(d / "features.npz", X=X[keep], names=np.array(["log_einc"] + names))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--criteria", type=Path, default=Path(__file__).with_name("criteria.json"))
    a = ap.parse_args(argv)
    crit_bytes = a.criteria.read_bytes()
    crit = json.loads(crit_bytes)
    nbins, max_rows, seed = int(crit["bins"]), int(crit["classifier_rows"]), int(crit["seed"])
    layers = read_binning(a.data / "binning_dataset_1_photons.xml")

    # floor and null from photons_1 (edges from the floor half only), stock from photons_2
    e1, s1 = read_h5(a.data / "dataset_1_photons_1.hdf5")
    check_order(layers, s1)
    energies = np.unique(np.round(e1, 3))
    if energies.size != 15:
        raise SystemExit(f"expected 15 incident energies in photons_1, found {energies.size}")
    perm = np.random.default_rng(seed).permutation(len(e1))
    fl, nu = perm[: len(e1) // 2], perm[len(e1) // 2:]
    f_floor = features(e1[fl], s1[fl], layers, np.float64)
    c_floor = energy_classes(e1[fl], energies)
    edges = make_edges(f_floor, c_floor, energies.size, nbins)
    names = sorted(f_floor)
    ref = a.out / "reference" / WORKFLOW
    write_arm(ref / "floor", f_floor, e1[fl], c_floor, edges, names, max_rows, seed)
    (ref / "reference.json").write_text(json.dumps({
        "task": WORKFLOW, "boundary": "incident photon at the calorimeter surface -> voxel energies",
        "mode": "shape", "min_count": 10, "stock_rate_per_slot": None,
        "classifier": {"file": "features.npz", "seed": seed},
        "criteria_sha256": hashlib.sha256(crit_bytes).hexdigest()}, indent=1))
    for dt in ("float32", "float64"):
        f = features(e1[nu], s1[nu], layers, np.dtype(dt).type)
        write_arm(a.out / "submissions" / "geant4_null" / WORKFLOW / dt, f, e1[nu], energy_classes(e1[nu], energies),
                  edges, names, max_rows, seed + 1)
    del s1

    e2, s2 = read_h5(a.data / "dataset_1_photons_2.hdf5")
    write_arm(ref / "stock", features(e2, s2, layers, np.float64), e2, energy_classes(e2, energies), edges, names,
              max_rows, seed + 2)
    del s2

    for sid, model in SUBMISSIONS.items():
        p = a.data / f"ds1-photons_{sid}.hdf5"
        if not p.is_file():
            print(f"missing {p.name} ({model}): skipped")
            continue
        e, s = read_h5(p)
        try:
            c = energy_classes(e, energies)
        except ValueError as err:
            print(f"{model}: {err}; skipped")
            continue
        for dt in ("float32", "float64"):
            write_arm(a.out / "submissions" / model / WORKFLOW / dt, features(e, s, layers, np.dtype(dt).type), e, c,
                      edges, names, max_rows, seed + 3)
        print(f"{model}: {len(e)} showers")
    print(f"{len(edges)} histograms per arm; criteria sha256 {hashlib.sha256(crit_bytes).hexdigest()[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
