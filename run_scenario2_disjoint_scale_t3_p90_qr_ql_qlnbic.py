from __future__ import annotations

import sys

import run_scenario2_disjoint_scale_t3_qr_ql_qlnbic as experiment


if __name__ == "__main__":
    extra_args = sys.argv[1:]
    if "--output-dir" not in extra_args:
        extra_args = ["--output-dir", "outputs/scenario2_disjoint_scale_t3_p90_qr_ql_qlnbic", *extra_args]
    sys.argv = [sys.argv[0], "--noise", "t3", "--p", "90", *extra_args]
    experiment.main()
