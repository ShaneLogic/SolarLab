"""Read a committed transient without importing or invoking its solver.

python examples/research_1d/read_transient.py --store /absolute/local/store --run-id completed
The read command succeeds for a readable partial history too; its numerical
outcome and failed attempt remain explicit in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from solarlab.experiments.transient_results import read_transient_result
from solarlab.io.run_store import RunStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id")
    args = parser.parse_args()
    with RunStore(args.store.expanduser().resolve(), read_only=True) as store:
        loaded = read_transient_result(store, args.run_id, attempt_id=args.attempt_id)
    arrays, diagnostics = loaded["arrays"], loaded["diagnostics"]
    print(json.dumps({"run_id": args.run_id, "attempt_id": loaded["document"]["attempt_id"],
        "terminal_state": loaded["run_state"], "numerical_outcome": loaded["document"]["outcome"],
        "physical_certificate_status": loaded["document"]["physical_certificate_status"],
        "provenance_status": loaded["provenance"]["status"],
        "requested_times_s": arrays["requested_times_s"].tolist(), "accepted_times_s": arrays["accepted_times_s"].tolist(),
        "current_times_s": arrays["accepted_times_s"][1:].tolist(),
        "terminal_conduction_current_density_A_m2": arrays["terminal_conduction_current_density_A_m2"].tolist(),
        "terminal_displacement_current_density_A_m2": arrays["terminal_displacement_current_density_A_m2"].tolist(),
        "failed_attempts": [{key: a[key] for key in ("start_s", "end_s", "accepted", "termination", "error")}
                            for a in diagnostics["attempts"] if not a["accepted"]],
        "arrays": {name: {"shape": list(value.shape), "dtype": value.dtype.str,
            "unit": loaded["manifest"]["payloads"][name]["metadata"]["unit"],
            "word_sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in arrays.items()}},
        indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
