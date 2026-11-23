"""Research extensions for the public SolarLab interfaces.

Provides:
* ``solarlab_research.precision``: double-double arithmetic (``DD``, ``DDArray``,
  ``CompensatedArithmetic``), compensated local two-sided interface carrier
  kernels, and reference-increment / affine input-lift state codecs.
* ``solarlab_research.cbo_scan``: public 1D conduction/valence-band-offset (CBO)
  scan orchestration pinned to pre-``P08`` models per ``D20``.
* ``solarlab_research.r1``: frozen V53 certificate limits, 13-channel operator
  schema, IDA ``R1-L2`` exclusion, immutable ``ReleaseBundle`` provenance and
  non-relabeling historical readers, controls ``A-D`` verification, and
  conservative ``4/8/16`` short-window public-interface verification.
* ``solarlab_research.scripts``: module CLI entrypoints invoked via
  ``python -m solarlab_research.scripts.<name>``.
"""

from solarlab_research import cbo_scan, precision, r1, scripts

__all__ = (
    "cbo_scan",
    "precision",
    "r1",
    "scripts",
)
