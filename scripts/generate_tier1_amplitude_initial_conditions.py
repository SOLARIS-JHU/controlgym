#!/usr/bin/env python3
"""Create Tier 1 amplitude-shifted initial-condition files.

This script reads the frozen Tier 0 held-out initial conditions
(<pde>/tier0_heldout_id/tier0_initial_conditions.npz), scales them by fixed
alpha values, and writes IC-only NPZ files under each PDE folder.

Tier 0 ICs are sampled from seeds disjoint from the training seed pool (see
tier0_seed_registry.yaml), which keeps this amplitude-shift eval set free of
the seed 999/123 contamination present in the old
<pde>/dpc/<pde>_dataset_zero_500.npz sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


DATA_ROOT = Path("/home/pkhuran3/scratchjdrgona1/pkhuran3/PDEControlDPC/data")
ALPHAS = (0.5, 0.75, 1.0, 1.5, 2.0)
# ALPHAS = (0.5, 1.5)
N_IC = 100
OUTPUT_SUBDIR = "tier1_amplitude_ic"


def stable_json(value) -> str:
    """Serialize to JSON with sorted keys so equal values always produce the same string."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def alpha_token(alpha: float) -> str:
    """Format an alpha value as a filesystem-safe token, e.g. 1.5 -> '1p50'."""
    return f"{alpha:.2f}".replace(".", "p")


def scalar_value(npz, key: str):
    """Read a scalar or array field from an NPZ archive, unwrapping 0-d arrays."""
    if key not in npz.files:
        return None
    value = npz[key]
    return value.item() if value.shape == () else value.tolist()


def json_value(npz, key: str):
    """Read an NPZ field and parse it as JSON, falling back to the raw value."""
    value = scalar_value(npz, key)
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def control_lifting_matrix(npz) -> tuple[str, np.ndarray]:
    """Find and return the control-to-state lifting matrix under whichever key it was saved as."""
    for key in ("B2", "b2", "control_sup", "control_lifting_map"):
        if key not in npz.files:
            continue
        matrix = npz[key]
        if matrix.shape == () and matrix.dtype == object and matrix.item() is None:
            continue
        return key, np.asarray(matrix).copy()
    raise KeyError("Source NPZ is missing required B2/b2/control_sup lifting matrix.")


def find_config(pde_dir: Path) -> Path:
    """Locate a PDE's linear or nonlinear DPC YAML config in its directory."""
    for name in (
        "linearPDE_data_gen.dpc.yaml",
        "nonlinearPDE_data_gen.dpc.yaml",
        # "linearPDE_data_gen_dpc.yaml",
        # "nonlinearPDE_data_gen_dpc.yaml",
    ):
        path = pde_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No DPC YAML found in {pde_dir}")


def load_config(path: Path) -> dict:
    """Load and validate a DPC YAML config file as a dict."""
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def validate_source(
    source_path: Path,
    config: dict,
    init_states: np.ndarray,
    lifting_matrix: np.ndarray,
) -> None:
    """Check that the source ICs and lifting matrix match the expected shapes and config."""
    if init_states.ndim != 3 or init_states.shape[1] != 1:
        raise ValueError(
            f"{source_path} has init_states shape {init_states.shape}; "
            "expected (N, 1, n_state)."
        )
    if init_states.shape[0] < N_IC:
        raise ValueError(f"{source_path} has {init_states.shape[0]} ICs; need {N_IC}.")

    n_state = config.get("env_kwargs", {}).get("n_state")
    if n_state is not None and int(n_state) != init_states.shape[-1]:
        raise ValueError(
            f"{source_path} has n_state={init_states.shape[-1]}, "
            f"but YAML has env_kwargs.n_state={n_state}."
        )
    if lifting_matrix.ndim != 2 or lifting_matrix.shape[0] != init_states.shape[-1]:
        raise ValueError(
            f"{source_path} has lifting matrix shape {lifting_matrix.shape}; "
            f"expected (n_state, n_ctrl) with n_state={init_states.shape[-1]}."
        )


def plot_init_state_comparison(npz_path: Path, count: int) -> Path:
    """Plot base vs. alpha-scaled ICs for a written NPZ file and save a PNG alongside it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(npz_path, allow_pickle=True) as data:
        base = data["base_init_states"][:, 0, :]
        scaled = data["init_states"][:, 0, :]
        alpha = float(data["alpha"])

    count = min(count, base.shape[0])
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.4 * count), squeeze=False)
    x = np.arange(base.shape[1])
    for idx, ax in enumerate(axes[:, 0]):
        ax.plot(x, base[idx], label="base", linewidth=1.5)
        ax.plot(x, scaled[idx], label=f"scaled alpha={alpha:g}", linewidth=1.5)
        ax.set_title(f"IC {idx}")
        ax.set_xlabel("state index")
        ax.legend()

    fig.tight_layout()
    plot_path = npz_path.with_name(f"{npz_path.stem}_comparison.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def write_scaled_ics(
    source_path: Path,
    overwrite: bool,
    dry_run: bool,
    plot: bool,
    plot_count: int,
) -> list[Path]:
    """Load one PDE's Tier 0 ICs and write an alpha-scaled NPZ for each ALPHAS value."""
    pde_dir = source_path.parents[1]
    pde = pde_dir.name
    config_path = find_config(pde_dir)
    config = load_config(config_path)
    output_dir = pde_dir / OUTPUT_SUBDIR
    written_paths = []

    with np.load(source_path, allow_pickle=True) as data:
        raw_init_states = data["init_states"][:N_IC]
        if raw_init_states.ndim == 2:
            raw_init_states = raw_init_states[:, None, :]
        init_states = raw_init_states.copy()
        source_seeds = np.asarray(data["seeds"])[:N_IC].copy()
        lifting_key, lifting_matrix = control_lifting_matrix(data)
        validate_source(source_path, config, init_states, lifting_matrix)

        source_env_id = scalar_value(data, "env_id")
        config_env_id = config.get("env_id")
        if source_env_id is not None and config_env_id is not None and source_env_id != config_env_id:
            raise ValueError(
                f"{source_path} has env_id={source_env_id}, "
                f"but {config_path} has env_id={config_env_id}."
            )

        env_id = source_env_id or config_env_id or pde_dir.name
        source_metadata = {
            "tier0_init_seed": scalar_value(data, "init_seed"),
            "tier0_seeds": source_seeds.tolist(),
            "tier0_generator_config_hash": scalar_value(data, "generator_config_hash"),
            "tier0_generator_script": scalar_value(data, "generator_script"),
            "lifting_key": lifting_key,
        }

    for alpha in ALPHAS:
        token = alpha_token(alpha)
        output_path = output_dir / f"{env_id}_init_states_alpha_{token}_{N_IC}.npz"

        print(f"{pde}: alpha={alpha} -> {output_path}")
        if dry_run:
            continue
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it.")

        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            init_states=alpha * init_states,
            base_init_states=init_states,
            B2=lifting_matrix,
            source_indices=np.arange(N_IC, dtype=np.int64),
            source_seeds=source_seeds,
            alpha=np.array(alpha, dtype=np.float64),
            N=np.array(N_IC, dtype=np.int64),
            pde=np.array(pde),
            env_id=np.array(env_id),
            source_npz=np.array(str(source_path.resolve())),
            config_path=np.array(str(config_path.resolve())),
            env_kwargs_json=np.array(stable_json(config.get("env_kwargs", {}))),
            source_metadata_json=np.array(stable_json(source_metadata)),
        )
        written_paths.append(output_path)
        if plot:
            plot_path = plot_init_state_comparison(output_path, plot_count)
            print(f"  plot: {plot_path}")

    return written_paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Tier 1 amplitude-shift IC generator."""
    parser = argparse.ArgumentParser(
        description="Scale frozen Tier 0 held-out initial conditions for Tier 1 amplitude-shift tests."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Find all PDEs with Tier 0 held-out ICs and write scaled Tier 1 IC files for each."""
    args = parse_args()
    data_root = args.data_root.expanduser()
    source_paths = sorted(data_root.glob("*/tier0_heldout_id/tier0_initial_conditions.npz"))
    if not source_paths:
        raise FileNotFoundError(f"No Tier 0 held-out IC files found under {data_root}")
    if args.plot_count <= 0:
        raise ValueError("--plot-count must be positive.")

    written_paths = []
    for source_path in source_paths:
        written_paths.extend(
            write_scaled_ics(
                source_path,
                args.overwrite,
                args.dry_run,
                args.plot,
                args.plot_count,
            )
        )

    if args.dry_run:
        print("Dry run only; no files written.")
    else:
        print(f"Wrote {len(written_paths)} files.")


if __name__ == "__main__":
    main()
