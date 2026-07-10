#!/usr/bin/env python3
"""Generate Tier 2 additive noisy initial conditions (Option 2, secondary).

x_noisy = x0 + (delta_target * ||x0||_2 / sqrt(n_state)) * standard_normal(n_state),
one draw per (level, ic) from np.random.default_rng(noise_seed), noise_seed shared
with gen_tier2_burnin_ics.py (NOISE_SEED_BASE + ic_index). No env is constructed;
this script has no controlgym import and no torch dependency.

Run directly from the repository root (or anywhere):
  python scripts/gen_tier2_additive_ics.py --pde burgers
  python scripts/gen_tier2_additive_ics.py --all
  python scripts/gen_tier2_additive_ics.py --pde burgers --level 1e-2 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import _tier2_common as common  # noqa: E402


def additive_state(x0: np.ndarray, delta_target: float, noise_seed: int, n_state: int) -> np.ndarray:
    rng = np.random.default_rng(noise_seed)
    sigma = delta_target * np.linalg.norm(x0) / np.sqrt(n_state)
    return x0 + sigma * rng.standard_normal(n_state)


def write_level_npz(out_path: Path, tier0: dict, level_tag: str, delta_target: float,
                     sigma_median: float, perturbed: np.ndarray) -> None:
    payload_for_hash = {
        "pde": tier0["pde"],
        "option": "additive",
        "level_tag": level_tag,
        "delta_target": delta_target,
        "noise_seed_base": common.NOISE_SEED_BASE,
        "tier0_generator_config_hash": tier0["generator_config_hash"],
    }
    digest = hashlib.sha256(common.stable_json(payload_for_hash).encode("utf-8")).hexdigest()

    common.savez_deterministic(
        out_path,
        init_states=perturbed,
        seeds=tier0["seeds"],
        B2=tier0["B2"],
        pde=np.array(tier0["pde"]),
        env_id=np.array(tier0["env_id"]),
        N=np.array(tier0["N"], dtype=np.int64),
        init_seed=np.array(tier0["init_seed"], dtype=np.int64),
        config_path=np.array(str(tier0["path"].resolve())),
        generator_script=np.array("gen_tier2_additive_ics.py"),
        generator_config_hash=np.array(digest),
        env_kwargs_json=np.array(common.stable_json(tier0["env_kwargs"])),
    )


def generate_pde(pde: str, data_root: Path, levels: list[str], out_root: Path):
    tier0 = common.load_tier0(data_root, pde)
    med_norm = common.compute_med_norm(tier0["init_states"])
    n_state = tier0["env_kwargs"]["n_state"]
    init_states = tier0["init_states"]
    N = tier0["N"]

    registry_entries = []
    written = []
    for level_tag in levels:
        delta_target = common.DELTA_TARGETS[level_tag]
        perturbed = np.empty_like(init_states)
        sigmas = np.empty(N)
        for i in range(N):
            noise_seed = common.NOISE_SEED_BASE + i
            x0 = init_states[i]
            sigmas[i] = delta_target * np.linalg.norm(x0) / np.sqrt(n_state)
            perturbed[i] = additive_state(x0, delta_target, noise_seed, n_state)

        deltas = np.linalg.norm(perturbed - init_states, axis=1) / np.linalg.norm(init_states, axis=1)
        median, iqr = common.delta_stats(deltas)
        print(
            f"[{pde}][d{level_tag}] additive measured delta median={median:.4g} iqr={iqr:.4g} "
            f"target={delta_target:.4g}"
        )

        sigma_median = float(np.median(sigmas))
        out_path = out_root / pde / f"tier2_additive_d{level_tag}" / common.OUTPUT_FILE_NAME
        write_level_npz(out_path, tier0, level_tag, delta_target, sigma_median, perturbed)
        written.append(out_path)

        registry_entries.append({
            "pde": pde,
            "option": "additive",
            "level_tag": level_tag,
            "delta_target": delta_target,
            "cov_or_sigma": sigma_median,
            "K": None,
            "med_norm": med_norm,
            "n_state": n_state,
            "noise_seed_base": common.NOISE_SEED_BASE,
            "measured_delta_median": median,
            "measured_delta_iqr": iqr,
            "flag_factor3_miss": False,
        })

    return registry_entries, written


def verify_one(pde: str, data_root: Path, level: str) -> None:
    real_path = data_root / pde / f"tier2_additive_d{level}" / common.OUTPUT_FILE_NAME
    if not real_path.exists():
        raise SystemExit(f"--verify: expected existing output at {real_path}; generate it first.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        generate_pde(pde, data_root, [level], tmp_root)
        tmp_path = tmp_root / pde / f"tier2_additive_d{level}" / common.OUTPUT_FILE_NAME

        real_bytes = real_path.read_bytes()
        tmp_bytes = tmp_path.read_bytes()
        if real_bytes != tmp_bytes:
            print(
                f"VERIFY FAIL: {real_path} ({len(real_bytes)} bytes) != "
                f"regenerated ({len(tmp_bytes)} bytes)"
            )
            raise SystemExit(1)
        print(f"VERIFY PASS: [{pde}][d{level}] regenerated output is byte-identical to {real_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Tier 2 additive noisy initial conditions (Option 2)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pde", help="Single PDE name (must have a frozen Tier-0 npz).")
    group.add_argument("--all", action="store_true", help="Process every PDE with a frozen Tier-0 npz.")
    parser.add_argument("--data-root", type=Path, default=common.DATA_ROOT)
    parser.add_argument("--level", choices=list(common.ADDITIVE_LEVELS), default=None,
                         help="Restrict to one level (default: run both).")
    parser.add_argument(
        "--verify", action="store_true",
        help="Regenerate one (pde, level) cell into a temp dir and assert it is "
             "byte-identical to the existing output; exits nonzero on mismatch.",
    )
    args = parser.parse_args()
    if args.verify and args.all:
        parser.error("--verify requires a single --pde, not --all.")
    return args


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser()

    if args.verify:
        level = args.level or common.ADDITIVE_LEVELS[0]
        verify_one(args.pde, data_root, level)
        return

    pdes = common.discover_pdes(data_root) if args.all else [args.pde]
    levels = [args.level] if args.level else list(common.ADDITIVE_LEVELS)

    all_entries = []
    for pde in pdes:
        entries, written = generate_pde(pde, data_root, levels, data_root)
        all_entries.extend(entries)
        for out_path in written:
            print(f"[{pde}] wrote {out_path}")

    common.append_registry_entries(all_entries)
    print(f"Updated registry: {common.REGISTRY_PATH}")


if __name__ == "__main__":
    main()
