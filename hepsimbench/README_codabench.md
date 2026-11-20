# Fast-simulation benchmark: three metrics

Each submission is one commit of a fast simulation, scored against the stock (full) simulation on the same
generated events, in float32 and float64:

- **Validity**: per histogram, chi2/ndf of the fast arm against stock, flagged when chi2/ndf > max(2, 3 x floor) and
  p < 1e-3, where the floor is stock with another seed against stock; yields pass when |ratio - 1| - 3 se is within
  the tolerance. Validity = (1 - flagged fraction) x (fraction of yields that pass).
- **Throughput multiplier**: the fast arm's warm rate over the stock rate per single-core grid slot at the same
  boundary (self-reported, labelled with the device).
- **Classifier score** (where the reference has per-object features): how well a fixed classifier tells the arms
  apart; 1 = indistinguishable.

Submission layout: `<workflow>/<dtype>/{hists.json, bench.json[, features.npz]}`. The scorer is
`hepsimbench.score` (NumPy only).
