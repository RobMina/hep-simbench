# CaloChallenge Dataset 1 photons: fast simulations against Geant4

The first public fast-vs-full chain scored by hep-simbench. All inputs are public (CC-BY-4.0):

| Input | Source | Role |
|---|---|---|
| `dataset_1_photons_2.hdf5` | Zenodo [10.5281/zenodo.8099322](https://doi.org/10.5281/zenodo.8099322) | stock (Geant4, the evaluation file) |
| `dataset_1_photons_1.hdf5`, first half | same record | floor (Geant4 with other seeds); the histogram edges come from it alone |
| `dataset_1_photons_1.hdf5`, second half | same record | `geant4_null`: Geant4 scored as a fast arm, which must pass |
| `ds1-photons_<id>.hdf5`, 15 files | Zenodo [10.5281/zenodo.15961728](https://doi.org/10.5281/zenodo.15961728) | fast arms: the samples submitted to the challenge (CaloDiffusion, CaloINN, Calo-VQ, CaloScore x3, CaloFlow teacher and student, CaloMan, BoloGAN, CaloShower2GAN, CaloShower3GAN, CaloVAE+INN, CaloForest, CaloGraph) |
| `binning_dataset_1_photons.xml` | [CaloChallenge/homepage](https://github.com/CaloChallenge/homepage) `code/` | voxel layout: 5 layers, 368 voxels |

Cite the dataset records and the challenge write-up (Rep. Prog. Phys. 88 116201, arXiv:2410.21611) with any result.

## Design

Per shower: E_tot / E_inc and, per layer, the energy fraction, the energy-weighted mean and rms voxel radius, the
energy-weighted mean x and y for layers with alpha bins, and the occupied-voxel fraction (`build.py`). Each feature
is histogrammed per incident-energy class (15 energies, 256 MeV to 4.2 TeV) and over all energies, in 40
equal-probability bins from the floor, with every edge moved off the values the floor takes (sparsity is
discrete, and an empty layer's fraction is exactly 0), so float32 rounding cannot move a population across an edge. The scorer runs in `shape` mode against the floor, with the fixed
classifier on the per-shower feature vectors (`classifier_rows` rows per arm). Throughput is not scored: there
is no stock Geant4 rate per grid slot for this dataset (`stock_rate_per_slot: null`). The features are computed in
float32 and in float64. The submitted samples are float32 files, so the two legs test the scorer chain's precision,
not a model's.

These features are hep-simbench's own, a subset in the spirit of the challenge's high-level features. They are not
the official `evaluate.py` metrics, and the numbers are not comparable with the paper's tables.

## Criteria (declared before any file is read)

`criteria.json`, whose sha256 goes into every reference and record:
- the null arm (`geant4_null`) has validity >= 0.95 and cls_score >= 0.9 in both legs;
- every arm's float32 and float64 validities agree within 0.02;
- the fast arms are descriptive: validity and cls_score per model, with no pass line. Their ranking is compared
  with the challenge's classifier ranking, and the comparison is reported, not used as a criterion.

## Running

```bash
bash validation/calochallenge/fetch.sh DATA_DIR                 # 2.5 GB; every md5 checked against Zenodo
python validation/calochallenge/build.py DATA_DIR OUT_DIR        # needs h5py
python validation/calochallenge/summarize.py OUT_DIR             # record.json, table.md
```

One job of 2 CPU and 8 GB is enough (one 121,000 x 368 file in float64 is 0.36 GB). CI runs the same chain on toy
files (`toy.py`) on every push.

## Result

Not run yet.
