"""hep-simbench: validation and benchmarking of a fast simulation against its stock (full) simulation.

- :mod:`hepsimbench.score`: fast-vs-stock scoring (validity with a reseeded-stock floor, both dtypes, throughput per
  grid slot, optional classifier, peak, power, track-agreement and portability extensions); NumPy only.
- :mod:`hepsimbench.bench`: warm-rate timer (setup, warm rate, N10, device label, compile counts, fresh inputs).
- :mod:`hepsimbench.approximations`: declared approximations with levels, presets and a provenance record.
- :mod:`hepsimbench.stats`: robust residual statistics and most-probable-value fits.
- :mod:`hepsimbench.from_root`, :mod:`hepsimbench.codabench`, :mod:`hepsimbench.synthetic`: ROOT histogram
  conversion, Codabench bundles, and a synthetic example of the layout.
"""
__version__ = "0.1.0"
