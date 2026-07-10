#!/usr/bin/env python3
"""Generate Tier 2 burn-in noisy initial conditions (Option 1, primary).

For each frozen Tier-0 IC, runs K=10 zero-control steps under process noise
only (sensor_noise_cov=0.0) from a fresh env seeded by a per-IC noise_seed,
and saves the resulting raw state as a new, perturbed IC. Three levels per
PDE: d0 (cov=0, paired control / free evolution), d1e-2, d2e-1 (cov
calibrated to a nominal target relative perturbation).

Run directly from the repository root (or anywhere; paths are resolved
relative to this file):
  python scripts/gen_tier2_burnin_ics.py --pde burgers
  python scripts/gen_tier2_burnin_ics.py --all
  python scripts/gen_tier2_burnin_ics.py --pde burgers --level 1e-2 --verify
  python scripts/gen_tier2_burnin_ics.py --pde burgers --plot

Base ICs, env_kwargs, and npz structure are loaded from the frozen Tier-0
file exactly as written by generate_tier0_initial_conditions.py; neither
that script nor any controlgym env source is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (_SCRIPTS_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import controlgym as gym  # noqa: E402
from linearPDE_data_gen import _resolve_env_target_state  # noqa: E402

import _tier2_common as common  # noqa: E402


def cov_for_delta(delta_target: float, med_norm: float, n_state: int, K: int) -> float:
    """Locked first-order calibration: cov = (delta_target * med_norm)^2 / (n_state * K)."""
    if delta_target == 0.0:
        return 0.0
    return (delta_target * med_norm) ** 2 / (n_state * K)


def run_level(pde: str, tier0: dict, level_tag: str, med_norm: float):
    """Run K burn-in steps for every Tier-0 IC at one noise level; return the perturbed
    states plus the cov and env_kwargs actually used (for the npz/registry record)."""
    delta_target = common.DELTA_TARGETS[level_tag]
    env_id = tier0["env_id"]
    n_state = tier0["env_kwargs"]["n_state"]
    cov = cov_for_delta(delta_target, med_norm, n_state, common.K_BURNIN)

    base_resolved = _resolve_env_target_state(dict(tier0["env_kwargs"]))
    env_kwargs = dict(base_resolved)
    env_kwargs["process_noise_cov"] = cov
    env_kwargs["sensor_noise_cov"] = 0.0

    print(
        f"[{pde}][d{level_tag}] env_kwargs diff vs Tier-0 (exactly 2 fields): "
        f"process_noise_cov {base_resolved.get('process_noise_cov')!r} -> {cov!r}, "
        f"sensor_noise_cov {base_resolved.get('sensor_noise_cov')!r} -> 0.0"
    )

    init_states = tier0["init_states"]
    N = tier0["N"]
    perturbed = np.empty_like(init_states)
    for i in range(N):
        start = time.perf_counter()
        noise_seed = common.NOISE_SEED_BASE + i
        env = gym.make(env_id, **env_kwargs)
        _, info = env.reset(seed=noise_seed, state=init_states[i])
        action = np.zeros(env.n_action)
        for _ in range(common.K_BURNIN):
            _, _, _, _, info = env.step(action)
        perturbed[i] = info["state"]
        elapsed = time.perf_counter() - start
        print(f"[{pde}][d{level_tag}] ic {i}/{N} took {elapsed:.3f}s")

    return perturbed, cov, env_kwargs


def write_level_npz(out_path: Path, tier0: dict, level_tag: str, delta_target: float,
                     cov: float, med_norm: float, perturbed: np.ndarray, env_kwargs_used: dict) -> None:
    payload_for_hash = {
        "pde": tier0["pde"],
        "option": "burnin",
        "level_tag": level_tag,
        "delta_target": delta_target,
        "process_noise_cov": cov,
        "K": common.K_BURNIN,
        "med_norm": med_norm,
        "env_kwargs": env_kwargs_used,
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
        generator_script=np.array("gen_tier2_burnin_ics.py"),
        generator_config_hash=np.array(digest),
        env_kwargs_json=np.array(common.stable_json(env_kwargs_used)),
    )


def plot_comparison(pde: str, tier0: dict, level_perturbed: dict, level_deltas: dict, npz_dir: Path) -> Path:
    """Save one figure: fixed ICs 0,1,2, one subplot each, overlaying tier0 vs d0 vs
    every other computed level on the spatial grid. Convention: schrodinger's state is
    [Re(u); Im(u)] -> plot |u|; wave's state is [displacement; velocity] -> plot
    displacement (first field); every other PDE plots the state as-is."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_state = tier0["env_kwargs"]["n_state"]

    def field(x):
        half = n_state // 2
        if pde == "schrodinger":
            return np.abs(x[:half] + 1j * x[half:])
        if pde == "wave":
            return x[:half]
        return x

    labels = {"0": "d0 (free evo)"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, ax in enumerate(axes):
        x0 = field(tier0["init_states"][idx])
        grid = np.arange(len(x0))
        ax.plot(grid, x0, label="tier0")
        title_bits = []
        for level_tag, perturbed in level_perturbed.items():
            label = labels.get(level_tag, f"d{level_tag}")
            ax.plot(grid, field(perturbed[idx]), label=label)
            title_bits.append(f"{label}={level_deltas[level_tag][idx]:.3g}")
        ax.set_title(f"IC {idx}: " + ", ".join(title_bits), fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()

    out_path = npz_dir / "tier2_burnin_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_pde(pde: str, data_root: Path, levels: list[str], out_root: Path, plot: bool = False):
    tier0 = common.load_tier0(data_root, pde)
    med_norm = common.compute_med_norm(tier0["init_states"])

    # d0 is always computed first: it is the paired control every other level's
    # measured delta is defined against, regardless of which levels were requested.
    ctrl_perturbed, ctrl_cov, ctrl_env_kwargs = run_level(pde, tier0, "0", med_norm)
    level_perturbed = {"0": ctrl_perturbed}
    level_deltas = {"0": np.zeros(tier0["N"])}

    registry_entries = []
    written = []
    for level_tag in levels:
        if level_tag == "0":
            perturbed, cov, env_kwargs_used = ctrl_perturbed, ctrl_cov, ctrl_env_kwargs
        else:
            perturbed, cov, env_kwargs_used = run_level(pde, tier0, level_tag, med_norm)

        ctrl_norms = np.linalg.norm(ctrl_perturbed, axis=1)
        deltas = np.linalg.norm(perturbed - ctrl_perturbed, axis=1) / ctrl_norms
        level_perturbed[level_tag] = perturbed
        level_deltas[level_tag] = deltas
        median, iqr = common.delta_stats(deltas)
        delta_target = common.DELTA_TARGETS[level_tag]

        flagged = False
        if delta_target > 0:
            ratio = median / delta_target
            flagged = ratio > 3.0 or ratio < (1.0 / 3.0)
            status = "FLAG >3x miss" if flagged else "ok"
            print(
                f"[{pde}][d{level_tag}] measured delta median={median:.4g} iqr={iqr:.4g} "
                f"target={delta_target:.4g} ratio={ratio:.3g} [{status}]"
            )
        else:
            print(f"[{pde}][d{level_tag}] control level; measured delta median={median:.4g} (expect 0)")

        out_path = out_root / pde / f"tier2_burnin_d{level_tag}" / common.OUTPUT_FILE_NAME
        write_level_npz(out_path, tier0, level_tag, delta_target, cov, med_norm, perturbed, env_kwargs_used)
        written.append(out_path)

        registry_entries.append({
            "pde": pde,
            "option": "burnin",
            "level_tag": level_tag,
            "delta_target": delta_target,
            "cov_or_sigma": cov,
            "K": common.K_BURNIN,
            "med_norm": med_norm,
            "n_state": tier0["env_kwargs"]["n_state"],
            "noise_seed_base": common.NOISE_SEED_BASE,
            "measured_delta_median": median,
            "measured_delta_iqr": iqr,
            "flag_factor3_miss": flagged,
        })

    if plot:
        # Same folder as the last npz written this run, whichever --level(s) were requested.
        plot_path = plot_comparison(pde, tier0, level_perturbed, level_deltas, out_path.parent)
        print(f"[{pde}] wrote {plot_path}")

    return registry_entries, written


def verify_one(pde: str, data_root: Path, level: str) -> None:
    real_path = data_root / pde / f"tier2_burnin_d{level}" / common.OUTPUT_FILE_NAME
    if not real_path.exists():
        raise SystemExit(f"--verify: expected existing output at {real_path}; generate it first.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        generate_pde(pde, data_root, [level], tmp_root)
        tmp_path = tmp_root / pde / f"tier2_burnin_d{level}" / common.OUTPUT_FILE_NAME

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
        description="Generate Tier 2 burn-in noisy initial conditions (Option 1)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pde", help="Single PDE name (must have a frozen Tier-0 npz).")
    group.add_argument("--all", action="store_true", help="Process every PDE with a frozen Tier-0 npz.")
    parser.add_argument("--data-root", type=Path, default=common.DATA_ROOT)
    parser.add_argument("--level", choices=list(common.BURNIN_LEVELS), default=None,
                         help="Restrict to one level (default: run all three).")
    parser.add_argument(
        "--verify", action="store_true",
        help="Regenerate one (pde, level) cell into a temp dir and assert it is "
             "byte-identical to the existing output; exits nonzero on mismatch.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save tier2_burnin_comparison.png (ICs 0,1,2 vs their burn-in states) into the "
             "last level's npz folder (data/<pde>/tier2_burnin_d<level>/).",
    )
    args = parser.parse_args()
    if args.verify and args.all:
        parser.error("--verify requires a single --pde, not --all.")
    return args


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser()

    if args.verify:
        level = args.level or common.BURNIN_LEVELS[0]
        verify_one(args.pde, data_root, level)
        return

    pdes = common.discover_pdes(data_root) if args.all else [args.pde]
    levels = [args.level] if args.level else list(common.BURNIN_LEVELS)

    all_entries = []
    for pde in pdes:
        entries, written = generate_pde(pde, data_root, levels, data_root, plot=args.plot)
        all_entries.extend(entries)
        for out_path in written:
            print(f"[{pde}] wrote {out_path}")

    common.append_registry_entries(all_entries)
    print(f"Updated registry: {common.REGISTRY_PATH}")


if __name__ == "__main__":
    main()
