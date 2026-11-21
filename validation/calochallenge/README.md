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

## Result (2026-11-21, hep-simbench `1d054ae`, one 2-CPU job, 6 min)

**Both declared checks pass.** The Geant4 null arm has validity 0.997 (1 of 363 histograms flagged) and
classifier AUC 0.499 (cls_score 1.000), and every arm's float32 and float64 validities agree within 0.02 (they agree
to 0.003 or better). `record.json` holds every arm's numbers; criteria sha256 `d456696a...`.

| Arm | validity f32 | validity f64 | flagged / hists (f64) | cls_score (f64) | AUC (f64) |
|---|---|---|---|---|---|
| geant4_null | 0.997 | 0.997 | 1 / 363 | 1.000 | 0.499 |
| CaloFlow_teacher | 0.636 | 0.636 | 132 / 363 | 0.697 | 0.652 |
| CaloScore | 0.802 | 0.802 | 72 / 363 | 0.679 | 0.661 |
| CaloFlow_student | 0.540 | 0.540 | 167 / 363 | 0.675 | 0.663 |
| CaloGraph | 0.231 | 0.231 | 279 / 363 | 0.644 | 0.678 |
| CaloScore_distilled | 0.628 | 0.628 | 135 / 363 | 0.616 | 0.692 |
| CaloINN | 0.700 | 0.700 | 109 / 363 | 0.597 | 0.701 |
| CaloScore_single-shot | 0.397 | 0.397 | 219 / 363 | 0.462 | 0.769 |
| CaloDiffusion | 0.669 | 0.672 | 119 / 363 | 0.317 | 0.842 |
| BoloGAN | 0.270 | 0.270 | 265 / 363 | 0.049 | 0.975 |
| CaloForest | 0.019 | 0.019 | 356 / 363 | 0.028 | 0.986 |
| CaloVAE+INN | 0.303 | 0.303 | 253 / 363 | 0.026 | 0.987 |
| CaloMan | 0.196 | 0.196 | 292 / 363 | 0.012 | 0.994 |
| CaloShower2GAN | 0.347 | 0.347 | 237 / 363 | 0.002 | 0.999 |
| Calo-VQ | 0.157 | 0.157 | 306 / 363 | 0.002 | 0.999 |
| CaloShower3GAN | 0.342 | 0.342 | 239 / 363 | 0.002 | 0.999 |

Reading the table:
- The 15 submissions spread from almost indistinguishable by this classifier (CaloFlow teacher, CaloScore,
  CaloFlow student: AUC 0.65-0.66) to almost fully separable (several GANs and Calo-VQ: AUC >= 0.99).
- Validity and the classifier rank differently: CaloScore has the best validity (0.80) and the second-best classifier
  score. CaloDiffusion has validity 0.67 but AUC 0.84. CaloGraph has AUC 0.68 but validity 0.23. Histograms per
  incident energy see shape errors that a classifier on all energies together can average away, and the other way
  round.
- 363 histograms per arm: 27 features (E_tot / E_inc; per layer the energy fraction, mean and rms radius, sparsity;
  x and y centroids in the two layers with alpha bins) over 15 energy classes and the inclusive set, where the
  floor has enough entries.

Not done: the declared cross-check against the challenge's own classifier ranking. The write-up's tables could not
be read (no HTML version; the PDF is too large for the fetch tool). The classifier here is hep-simbench's fixed
logistic regression on quadratic high-level features, not the challenge's neural-network classifiers on voxels, so
its AUCs are not comparable with the paper's tables in any case, and only the order could be compared.