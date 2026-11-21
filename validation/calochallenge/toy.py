"""Toy files in the CaloChallenge Dataset 1 photon format (no real showers), so CI runs build.py and summarize.py
end to end: a binning file with the Dataset 1 photon layout (5 layers, 368 voxels), photons_1/_2 and two submissions
(`ds1-photons_1.hdf5` drawn like Geant4, `ds1-photons_4.hdf5` with a narrower, shifted shower).

    python validation/calochallenge/toy.py DATA_DIR [--showers 3000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LAYERS = [(0, [0, 5, 10, 30, 50, 100, 200, 400, 600], 1),
          (1, [0, 1, 4, 7, 10, 15, 20, 30, 40, 50, 70, 90, 120, 150, 200, 300, 400], 10),
          (2, [0, 1, 4, 7, 10, 15, 20, 30, 40, 50, 70, 90, 120, 150, 200, 300, 400, 600, 800, 1000], 10),
          (3, [0, 50, 100, 200, 400, 600], 1),
          (12, [0, 100, 200, 400, 1000, 2000], 1)]
FRACTIONS = np.array([0.02, 0.35, 0.55, 0.05, 0.01])
ENERGIES = 256.0 * 2.0 ** np.arange(15)


def write_xml(path: Path) -> None:
    rows = "\n".join(f'    <Layer id="{i}" r_edges="{",".join(str(v) for v in r)}" n_bin_alpha="{na}"/>'
                     for i, r, na in LAYERS)
    path.write_text(f'<Bins>\n  <Particle name="photon" pid="22">\n{rows}\n  </Particle>\n</Bins>\n')


def showers(n: int, seed: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    einc = rng.choice(ENERGIES, n)
    out = []
    for (lid, r, na), frac in zip(LAYERS, FRACTIONS):
        r = np.asarray(r, float)
        rc = 0.5 * (r[1:] + r[:-1])
        w = np.exp(-rc / (25.0 * scale)) * np.diff(r ** 2)
        w = np.tile(w / w.sum(), na) / na
        e_layer = einc * frac * rng.lognormal(0.0, 0.1, n)
        out.append(rng.dirichlet(w * 50 + 1e-3, n) * e_layer[:, None])
    s = np.hstack(out).astype(np.float32)
    s[s < 1e-3 * einc[:, None]] = 0
    return einc[:, None].astype(np.float32), s


def write_h5(path: Path, einc, s) -> None:
    import h5py
    with h5py.File(path, "w") as f:
        f["incident_energies"], f["showers"] = einc, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data", type=Path)
    ap.add_argument("--showers", type=int, default=3000)
    a = ap.parse_args(argv)
    a.data.mkdir(parents=True, exist_ok=True)
    write_xml(a.data / "binning_dataset_1_photons.xml")
    n = a.showers
    write_h5(a.data / "dataset_1_photons_1.hdf5", *showers(2 * n, 1))
    write_h5(a.data / "dataset_1_photons_2.hdf5", *showers(n, 2))
    write_h5(a.data / "ds1-photons_1.hdf5", *showers(n, 3))
    write_h5(a.data / "ds1-photons_4.hdf5", *showers(n, 4, scale=0.7))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
