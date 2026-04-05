#!/usr/bin/env python3
"""Orchestrate log-banded PDE dataset generation and random-band merging.

Examples:
  .venv/bin/python generate_logband_dataset.py --config linearPDE_data_gen.default.yaml
  .venv/bin/python generate_logband_dataset.py --config nonlinearPDE_data_gen.default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


LINEAR_ENV_IDS = {"convection_diffusion_reaction", "wave", "schrodinger"}
NONLINEAR_ENV_IDS = {
    "allen_cahn",
    "burgers",
    "cahn_hilliard",
    "fisher",
    "ginzburg_landau",
    "korteweg_de_vries",
    "kuramoto_sivashinsky",
}
ROW_KEYS = ("solutions", "init_states", "controls", "controls_field", "R", "T")
PER_BAND_KEYS = {
    "random_signal_kwargs_json",
    "total_time_sec",
    "mean_time_per_traj_sec",
    "dataset_modes_json",
    "ctrl_mode",
}


def _log(message: str = "") -> None:
    print(message, flush=True)


def make_log_bands(n_steps: int, total_trajectories: int, n_bands: int = 5):
    nyquist = n_steps // 2
    r = (nyquist / 1.0) ** (1.0 / n_bands)
    bands = [(round(r**i, 2), round(r ** (i + 1), 2)) for i in range(n_bands)]
    base = total_trajectories // n_bands
    remainder = total_trajectories - base * n_bands
    traj_per_band = [base + (1 if i < remainder else 0) for i in range(n_bands)]
    return bands, traj_per_band


def _load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError(f"YAML config at {path} must contain a top-level mapping.")
    return config


def _require_config_value(config: dict, key: str):
    if key not in config or config[key] is None:
        raise ValueError(f"Config is missing required key: {key}")
    return config[key]


def _infer_generator_script(config_path: Path, config: dict, repo_root: Path) -> Path:
    filename = config_path.name.lower()
    env_id = config.get("env_id")

    if "nonlinear" in filename:
        script = repo_root / "nonlinearPDE_data_gen.py"
    elif "linear" in filename:
        script = repo_root / "linearPDE_data_gen.py"
    elif any(key in config for key in ("ppo_checkpoint_dir", "train_controller", "ppo_train_kwargs")):
        script = repo_root / "nonlinearPDE_data_gen.py"
    elif env_id in NONLINEAR_ENV_IDS:
        script = repo_root / "nonlinearPDE_data_gen.py"
    elif env_id in LINEAR_ENV_IDS:
        script = repo_root / "linearPDE_data_gen.py"
    else:
        raise ValueError(
            "Could not infer generator script from config path or env_id. "
            f"config={config_path}, env_id={env_id!r}"
        )

    if not script.is_file():
        raise FileNotFoundError(f"Generator script not found: {script}")
    return script


def _band1_modes(
    generator_script: Path,
    config: dict,
    include_ctrl_band1: bool,
) -> tuple[str, ...]:
    if generator_script.name != "nonlinearPDE_data_gen.py":
        return ("zero", "random", "ctrl")

    if not include_ctrl_band1:
        return ("zero", "random")

    if config.get("train_controller") or config.get("ppo_checkpoint_dir"):
        return ("zero", "random", "ctrl")

    raise ValueError(
        "--include-ctrl-band1 was requested for a nonlinear config, but neither "
        "train_controller: true nor ppo_checkpoint_dir is set."
    )


def _default_output_root(config: dict) -> Path:
    output_dir = Path(config.get("output_dir", "."))
    return output_dir / "logbands"


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"Output root exists and is not a directory: {output_root}")
        if not overwrite and any(output_root.iterdir()):
            raise FileExistsError(
                f"Output root is not empty: {output_root}. Pass --overwrite to reuse it."
            )
    output_root.mkdir(parents=True, exist_ok=True)


def _band_dir_name(index: int, lo: float, hi: float) -> str:
    return f"band_{index:02d}_{lo}_{hi}"


def _expected_dataset_paths(env_id: str, num_traj: int, output_dir: Path, modes: tuple[str, ...]) -> list[Path]:
    return [output_dir / f"{env_id}_dataset_{mode}_{num_traj}.npz" for mode in modes]


def _np_value_for_npz(value):
    if value is None:
        return np.array(None, dtype=object)
    return np.array(value)


def _scalar_item(array: np.ndarray):
    return array.item() if array.shape == () else array


def _string_scalar(array: np.ndarray) -> str:
    value = _scalar_item(array)
    if value is None:
        return "null"
    return str(value)


def _values_equal(lhs: np.ndarray, rhs: np.ndarray) -> bool:
    if lhs.shape != rhs.shape:
        return False
    if lhs.dtype == object or rhs.dtype == object:
        return np.array_equal(lhs, rhs)
    if np.issubdtype(lhs.dtype, np.number) and np.issubdtype(rhs.dtype, np.number):
        return np.allclose(lhs, rhs, equal_nan=True)
    return np.array_equal(lhs, rhs)


def _run_band(
    generator_script: Path,
    config_path: Path,
    band_dir: Path,
    env_id: str,
    num_traj: int,
    band_index: int,
    lo: float,
    hi: float,
    modes: tuple[str, ...],
) -> list[Path]:
    band_dir.mkdir(parents=True, exist_ok=True)

    _log(
        f"\nRunning band {band_index}: cycles_range=({lo}, {hi}) "
        f"dataset_modes={list(modes)} output_dir={band_dir}"
    )

    cmd = [
        sys.executable,
        str(generator_script),
        "--config",
        str(config_path),
        "--dataset-modes",
        *modes,
        "--random-signal-mode",
        "env",
        "--random-cycles-range",
        str(lo),
        str(hi),
        "--output-dir",
        str(band_dir),
    ]
    subprocess.run(cmd, check=True, cwd=generator_script.parent)

    expected_paths = _expected_dataset_paths(env_id, num_traj, band_dir, modes)
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Band {band_index} completed but expected files are missing: {missing_str}")

    _log(f"Band {band_index} saved files:")
    for path in expected_paths:
        _log(f"  {path}")

    return expected_paths


def _scan_band_files(
    random_paths: list[Path],
    expected_band_count: int,
) -> tuple[
    dict[str, tuple[tuple[int, ...], np.dtype]],
    dict[str, np.ndarray],
    list[int],
    list[str],
    list[float],
    list[float],
]:
    row_specs: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    invariant_metadata: dict[str, np.ndarray] = {}
    row_counts: list[int] = []
    band_random_kwargs_json: list[str] = []
    band_total_times: list[float] = []
    band_mean_times: list[float] = []
    file_keys: set[str] | None = None

    for path in random_paths:
        with np.load(path, allow_pickle=True) as data:
            current_keys = set(data.files)
            if file_keys is None:
                file_keys = current_keys
            elif current_keys != file_keys:
                raise ValueError(f"Mismatched keys in {path}: {sorted(current_keys)} != {sorted(file_keys)}")

            for key in ROW_KEYS:
                if key not in data.files:
                    raise KeyError(f"{path} is missing required row key: {key}")
                array = data[key]
                if key not in row_specs:
                    row_specs[key] = (array.shape[1:], array.dtype)
                else:
                    shape_suffix, dtype = row_specs[key]
                    if array.shape[1:] != shape_suffix or array.dtype != dtype:
                        raise ValueError(
                            f"Row key mismatch for {key} in {path}: "
                            f"shape={array.shape[1:]}, dtype={array.dtype}, "
                            f"expected shape={shape_suffix}, dtype={dtype}"
                        )

            row_count = data[ROW_KEYS[0]].shape[0]
            if row_count != expected_band_count:
                raise ValueError(
                    f"Expected {expected_band_count} rows per band, but {path} has {row_count} rows."
                )
            for key in ROW_KEYS[1:]:
                if data[key].shape[0] != row_count:
                    raise ValueError(f"Row count mismatch for key {key} in {path}")
            row_counts.append(row_count)

            band_random_kwargs_json.append(_string_scalar(data["random_signal_kwargs_json"]))
            band_total_times.append(float(_scalar_item(data["total_time_sec"])))
            band_mean_times.append(float(_scalar_item(data["mean_time_per_traj_sec"])))

            for key in data.files:
                if key in ROW_KEYS or key in PER_BAND_KEYS:
                    continue
                value = data[key]
                if key not in invariant_metadata:
                    invariant_metadata[key] = value
                elif not _values_equal(invariant_metadata[key], value):
                    raise ValueError(f"Invariant metadata mismatch for key {key} in {path}")

    return (
        row_specs,
        invariant_metadata,
        row_counts,
        band_random_kwargs_json,
        band_total_times,
        band_mean_times,
    )


def _merge_random_band_files(
    random_paths: list[Path],
    band_dirs: list[Path],
    bands: list[tuple[float, float]],
    num_traj: int,
    output_path: Path,
    shuffle_seed: int,
    config: dict,
) -> dict[str, object]:
    (
        row_specs,
        invariant_metadata,
        row_counts,
        band_random_kwargs_json,
        band_total_times,
        band_mean_times,
    ) = _scan_band_files(random_paths, expected_band_count=num_traj)

    total_rows = sum(row_counts)
    rng = np.random.default_rng(shuffle_seed)
    permutation = rng.permutation(total_rows)
    shuffled_destinations = np.empty(total_rows, dtype=np.int64)
    shuffled_destinations[permutation] = np.arange(total_rows, dtype=np.int64)

    combined = {
        key: np.empty((total_rows, *shape_suffix), dtype=dtype)
        for key, (shape_suffix, dtype) in row_specs.items()
    }

    source_dir_names = [str(path.name) for path in band_dirs]
    band_source_dir = np.empty(total_rows, dtype=f"<U{max(len(name) for name in source_dir_names)}")
    band_index = np.empty(total_rows, dtype=np.int64)
    band_cycles_range = np.empty((total_rows, 2), dtype=np.float64)

    start = 0
    for band_idx, (path, source_dir, count, cycles_range) in enumerate(
        zip(random_paths, source_dir_names, row_counts, bands),
        start=1,
    ):
        stop = start + count
        destinations = shuffled_destinations[start:stop]
        with np.load(path, allow_pickle=True) as data:
            for key in ROW_KEYS:
                combined[key][destinations] = data[key]
        band_source_dir[destinations] = source_dir
        band_index[destinations] = band_idx
        band_cycles_range[destinations] = np.asarray(cycles_range, dtype=np.float64)
        start = stop

    weighted_mean_time = float(np.average(np.asarray(band_mean_times), weights=np.asarray(row_counts)))
    random_kwargs_summary = json.dumps([json.loads(item) for item in band_random_kwargs_json], sort_keys=True)

    payload = dict(combined)
    payload.update(invariant_metadata)

    # Band 1's random export comes from a multi-mode run while later bands are random-only.
    # Normalize the merged metadata to reflect the merged file's random-only semantics.
    payload["env_id"] = np.array(str(_require_config_value(config, "env_id")))
    payload["ctrl_mode"] = _np_value_for_npz(config.get("ctrl_mode"))
    payload["export_mode"] = np.array("random")
    payload["dataset_modes_json"] = np.array(json.dumps(["random"]))
    payload["random_signal_kwargs_json"] = np.array(random_kwargs_summary)
    payload["total_time_sec"] = np.array(sum(band_total_times), dtype=np.float64)
    payload["mean_time_per_traj_sec"] = np.array(weighted_mean_time, dtype=np.float64)
    payload["band_index"] = band_index
    payload["band_cycles_range"] = band_cycles_range
    payload["band_source_dir"] = band_source_dir
    payload["band_counts"] = np.asarray(row_counts, dtype=np.int64)
    payload["band_total_time_sec"] = np.asarray(band_total_times, dtype=np.float64)
    payload["band_mean_time_per_traj_sec"] = np.asarray(band_mean_times, dtype=np.float64)
    payload["band_random_signal_kwargs_json"] = np.asarray(band_random_kwargs_json)

    np.savez_compressed(output_path, **payload)

    return {
        "output_path": output_path,
        "total_rows": total_rows,
        "row_shapes": {key: payload[key].shape for key in ROW_KEYS},
        "band_counts": row_counts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate log-banded PDE datasets and merge random-band outputs."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n-bands", type=int, default=5)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-ctrl-band1",
        action="store_true",
        help=(
            "For nonlinear configs, also generate ctrl during band 1. "
            "Requires train_controller: true or ppo_checkpoint_dir in the config. "
            "Linear configs already include ctrl by default."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent
    config_path = args.config.resolve()
    config = _load_yaml_config(config_path)

    env_id = str(_require_config_value(config, "env_id"))
    num_traj = int(_require_config_value(config, "num_traj"))
    env_kwargs = _require_config_value(config, "env_kwargs")
    if not isinstance(env_kwargs, dict):
        raise ValueError("'env_kwargs' in config must be a mapping.")
    n_steps = int(_require_config_value(env_kwargs, "n_steps"))

    if args.n_bands <= 0:
        raise ValueError("--n-bands must be a positive integer.")
    if num_traj <= 0:
        raise ValueError("'num_traj' in config must be a positive integer.")
    if n_steps <= 1:
        raise ValueError("'env_kwargs.n_steps' in config must be greater than 1.")

    generator_script = _infer_generator_script(config_path, config, repo_root)
    band1_modes = _band1_modes(
        generator_script=generator_script,
        config=config,
        include_ctrl_band1=args.include_ctrl_band1,
    )

    total_trajectories = num_traj * args.n_bands
    bands, counts = make_log_bands(n_steps=n_steps, total_trajectories=total_trajectories, n_bands=args.n_bands)
    if any(count != num_traj for count in counts):
        raise ValueError(
            f"Computed band trajectory counts {counts} do not match config num_traj={num_traj}."
        )

    output_root = args.output_root if args.output_root is not None else _default_output_root(config)
    _prepare_output_root(output_root, overwrite=args.overwrite)

    _log("Log-band generation configuration:")
    _log(f"  generator_script: {generator_script}")
    _log(f"  config: {config_path}")
    _log(f"  env_id: {env_id}")
    _log(f"  n_steps: {n_steps}")
    _log(f"  num_traj_per_band: {num_traj}")
    _log(f"  n_bands: {args.n_bands}")
    _log(f"  output_root: {output_root}")
    _log(f"  band_1_modes: {list(band1_modes)}")
    _log("  later_band_modes: ['random']")
    _log("  bands:")
    for index, ((lo, hi), count) in enumerate(zip(bands, counts), start=1):
        _log(f"    {index}: cycles_range=({lo}, {hi}) count={count}")

    random_paths: list[Path] = []
    band_dirs: list[Path] = []

    for band_index, (lo, hi) in enumerate(bands, start=1):
        band_dir = output_root / _band_dir_name(band_index, lo, hi)
        band_dirs.append(band_dir)
        modes = band1_modes if band_index == 1 else ("random",)
        saved_paths = _run_band(
            generator_script=generator_script,
            config_path=config_path,
            band_dir=band_dir,
            env_id=env_id,
            num_traj=num_traj,
            band_index=band_index,
            lo=lo,
            hi=hi,
            modes=modes,
        )
        random_paths.append(next(path for path in saved_paths if "_dataset_random_" in path.name))

    combined_output_path = output_root / "dataset_combined_logbands.npz"
    summary = _merge_random_band_files(
        random_paths=random_paths,
        band_dirs=band_dirs,
        bands=bands,
        num_traj=num_traj,
        output_path=combined_output_path,
        shuffle_seed=args.shuffle_seed,
        config=config,
    )

    _log("\nMerged random dataset summary:")
    _log(f"  total_trajectories: {summary['total_rows']}")
    _log(f"  per_band_counts: {summary['band_counts']}")
    _log("  array_shapes:")
    for key, shape in summary["row_shapes"].items():
        _log(f"    {key}: {shape}")
    _log(f"  saved: {summary['output_path']}")


if __name__ == "__main__":
    main()
