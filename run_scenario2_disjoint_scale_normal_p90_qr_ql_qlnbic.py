from __future__ import annotations

import sys

import run_scenario2_disjoint_scale_t3_qr_ql_qlnbic as experiment


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--noise", "normal", "--p", "90", *sys.argv[1:]]
    experiment.main()
