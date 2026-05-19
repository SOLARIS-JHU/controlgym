#!/usr/bin/env python3
"""Command-line dataset generation for linear PDE environments.

Run directly from the repository root:

1. With the default YAML config:
   `.venv/bin/python linearPDE_data_gen.py --config linearPDE_data_gen.default.yaml`

2. With YAML config plus CLI overrides:
   `.venv/bin/python linearPDE_data_gen.py --config linearPDE_data_gen.default.yaml --num-traj 300 --ctrl-mode lqr`

3. Without YAML, passing all key arguments on the command line:
   `.venv/bin/python linearPDE_data_gen.py --env-id wave --num-traj 300 --ctrl-mode lqg --n-state 200 --n-observation 200 --n-action 8 --n-steps 200 --process-noise-cov 0 --sensor-noise-cov 1e-8 --random-min-val -0.5 --random-max-val 0.5`

4. Save outputs to a specific directory:
   `.venv/bin/python linearPDE_data_gen.py --config linearPDE_data_gen.default.yaml --output-dir /tmp/linear_pde_runs`

Outputs written for each requested mode:
- `{env_id}_dataset_{mode}_{num_traj}.npz`
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from pprint import pformat
import time

import numpy as np
import yaml

import controlgym as gym


LINEAR_PDE_IDS = {"convection_diffusion_reaction", "wave", "schrodinger"}
VALID_CTRL_MODES = {"lqg", "lqr"}
DATASET_MODES = ("zero", "random", "ctrl")
DEFAULT_RANDOM_SIGNAL_KWARGS = {
    "signal_mode": "manual",
    "freq_range": (1, 3),
    "min_val": -1.0,
    "max_val": 1.0,
    "amp_range": (10, 20),
}
DEFAULT_RUN_CONFIG = {
    "ctrl_mode": "lqg",
    "dataset_modes": list(DATASET_MODES),
    "init_seed": 123,
    "noise_seed": 10_000,
    "sample_check_count": 5,
    "sample_check_seed": 42,
    "fill_nan": False,
    "output_dir": ".",
    "env_kwargs": {},
    "random_signal_kwargs": {
        "signal_mode": "manual",
        "freq_range": [1, 3],
        "min_val": -1.0,
        "max_val": 1.0,
        "amp_range": [10, 20],
        "num_modes": 4,
        "zero_mean": True,
        "per_dim": True,
    },
}

ENV_ARG_KEYS = (
    "n_steps",
    "domain_length",
    "integration_time",
    "sample_time",
    "process_noise_cov",
    "sensor_noise_cov",
    "n_state",
    "n_observation",
    "n_action",
    "control_sup_width",
    "Q_weight",
    "R_weight",
    "action_limit",
    "observation_limit",
    "reward_limit",
)

RANDOM_ARG_MAP = {
    "signal_mode": "random_signal_mode",
    "freq_range": "random_freq_range",
    "min_val": "random_min_val",
    "max_val": "random_max_val",
    "amp_range": "random_amp_range",
    "cycles_range": "random_cycles_range",
    "num_modes": "random_num_modes",
    "zero_mean": "random_zero_mean",
    "per_dim": "random_per_dim",
}


def _is_2d_array(x) -> bool:
    """Return True if x is a NumPy matrix-like 2D array."""
    return isinstance(x, np.ndarray) and x.ndim == 2


def _validate_dataset_modes(dataset_modes) -> tuple[str, ...]:
    dataset_modes = DATASET_MODES if dataset_modes is None else tuple(dataset_modes)
    if len(dataset_modes) == 0:
        raise ValueError("dataset_modes must contain at least one mode.")
    if len(set(dataset_modes)) != len(dataset_modes):
        raise ValueError("dataset_modes contains duplicate modes.")
    unknown = set(dataset_modes) - set(DATASET_MODES)
    if unknown:
        raise ValueError(f"Unknown dataset modes: {unknown}. Allowed: {DATASET_MODES}")
    return dataset_modes


def _validate_inputs(
    env_id: str,
    ctrl_mode: str,
    n_traj: int,
    dataset_modes,
) -> tuple[str, ...]:
    if env_id not in LINEAR_PDE_IDS:
        raise ValueError(
            f"{env_id} is nonlinear in controlgym. LQG/LQR are supported only for {LINEAR_PDE_IDS}."
        )
    if n_traj <= 0:
        raise ValueError("n_traj must be a positive integer.")
    dataset_modes = _validate_dataset_modes(dataset_modes)
    if "ctrl" in dataset_modes and ctrl_mode not in VALID_CTRL_MODES:
        raise ValueError("ctrl_mode must be 'lqg' or 'lqr' when 'ctrl' is requested.")
    return dataset_modes


def _make_envs(env_id: str, dataset_modes, **env_kwargs):
    env_kwargs = _resolve_env_target_state(env_kwargs)
    env_init = gym.make(env_id, **env_kwargs)
    envs = {mode: gym.make(env_id, **env_kwargs) for mode in dataset_modes}
    return env_init, envs


def _resolve_env_target_state(env_kwargs: dict) -> dict:
    """Expand scalar target_state values to full state vectors."""
    env_kwargs = dict(env_kwargs)
    if "target_state" not in env_kwargs:
        return env_kwargs

    target_state = np.asarray(env_kwargs["target_state"], dtype=float)
    if target_state.ndim == 0:
        env_kwargs["target_state"] = np.full(int(env_kwargs["n_state"]), float(target_state))
        return env_kwargs

    if target_state.shape != (int(env_kwargs["n_state"]),):
        raise ValueError(
            "env_kwargs.target_state must be a scalar or have shape "
            f"({env_kwargs['n_state']},). Got shape {target_state.shape}."
        )
    env_kwargs["target_state"] = target_state
    return env_kwargs


def _resolve_random_signal_kwargs(random_signal_kwargs: dict | None) -> dict:
    resolved = dict(DEFAULT_RANDOM_SIGNAL_KWARGS)
    if random_signal_kwargs is not None:
        resolved.update(dict(random_signal_kwargs))
    resolved.pop("env", None)
    resolved.pop("seed", None)
    return resolved


def _make_controllers(
    envs,
    dataset_modes,
    ctrl_mode: str,
    random_signal_kwargs: dict,
):
    controllers = {}

    if "zero" in dataset_modes:
        controllers["zero"] = gym.controllers.Zero(envs["zero"])

    if "random" in dataset_modes:
        controllers["random"] = gym.controllers.SmoothRandom(
            envs["random"],
            **random_signal_kwargs,
        )

    if "ctrl" in dataset_modes:
        env_ctrl = envs["ctrl"]
        ctrl = gym.controllers.LQG(env_ctrl) if ctrl_mode == "lqg" else gym.controllers.LQR(env_ctrl)
        if ctrl_mode == "lqg":
            if getattr(env_ctrl, "sensor_noise_cov", 0.0) <= 0:
                raise ValueError("LQG requires sensor_noise_cov > 0.")
            if not _is_2d_array(ctrl.gain_lqr):
                raise RuntimeError("LQG gain_lqr is invalid. Try different env/noise settings.")
            if not _is_2d_array(ctrl.gain_kf):
                raise RuntimeError(
                    "LQG gain_kf is invalid (not a matrix). "
                    "Try setting n_observation=n_state and positive process/sensor noise covariances."
                )
        controllers["ctrl"] = ctrl

    return controllers


def _sample_initial_state(env_init, seed: int) -> np.ndarray:
    _, info = env_init.reset(seed=seed)
    return info["state"].copy()


def _extract_shared_b2(*envs) -> np.ndarray:
    b2 = envs[0].B2.copy()
    for env in envs[1:]:
        if not np.allclose(b2, env.B2):
            raise RuntimeError(
                "Expected all requested linear PDE env clones to share the same B2 matrix."
            )
    return b2


def _select_action(mode: str, controller, current_state: np.ndarray, est_state: np.ndarray | None):
    if mode in {"zero", "random"}:
        return controller.select_action()
    if mode == "lqr":
        return controller.select_action(current_state)
    if mode == "lqg":
        return controller.select_action(est_state)
    raise ValueError(f"Unknown mode: {mode}")


def rollout_with_actions(env, controller, mode: str, x0: np.ndarray, noise_seed: int):
    """Run one rollout and return X, U, reward sum, and valid horizon T."""
    if mode == "random":
        controller.reset(seed=noise_seed)

    _, info = env.reset(seed=noise_seed, state=x0)
    current_state = info["state"]

    X = np.full((env.n_state, env.n_steps + 1), np.nan)
    U = np.full((env.n_action, env.n_steps), np.nan)
    X[:, 0] = current_state

    total_reward = 0.0
    est_state = current_state.copy() if mode == "lqg" else None
    use_kf = mode == "lqg" and _is_2d_array(getattr(controller, "gain_kf", None))

    for t in range(env.n_steps):
        u = _select_action(mode, controller, current_state, est_state)
        obs, reward, terminated, truncated, info = env.step(u)

        current_state = info["state"]
        X[:, t + 1] = current_state
        U[:, t] = u
        total_reward += reward

        if terminated or truncated:
            T = t + 1
            return X[:, : T + 1], U[:, : T], total_reward, T

        if mode == "lqg":
            if use_kf:
                est_state = controller.evolve_state_estimate(est_state, u, obs)
            else:
                est_state = current_state.copy()

    return X, U, total_reward, env.n_steps


def _compute_control_field(b2: np.ndarray, U: np.ndarray) -> np.ndarray:
    return b2 @ U


def _allocate_dataset_arrays(
    dataset_modes,
    n_traj: int,
    n_state: int,
    n_action: int,
    n_steps: int,
):
    return {
        "init_states": np.zeros((n_traj, n_state)),
        "X": {
            mode: np.full((n_traj, n_state, n_steps + 1), np.nan)
            for mode in dataset_modes
        },
        "U": {
            mode: np.full((n_traj, n_action, n_steps), np.nan)
            for mode in dataset_modes
        },
        "U_field": {
            mode: np.full((n_traj, n_state, n_steps), np.nan)
            for mode in dataset_modes
        },
        "R": {mode: np.zeros(n_traj) for mode in dataset_modes},
        "T": {mode: np.zeros(n_traj, dtype=int) for mode in dataset_modes},
    }


def _store_one_rollout(data, mode: str, b2: np.ndarray, k: int, X, U, R: float, T: int):
    data["X"][mode][k, :, : X.shape[1]] = X
    data["U"][mode][k, :, : U.shape[1]] = U
    F = _compute_control_field(b2, U)
    data["U_field"][mode][k, :, : F.shape[1]] = F
    data["R"][mode][k] = R
    data["T"][mode][k] = T


def generate_controlled_dataset(
    env_id: str,
    n_traj: int,
    ctrl_mode: str = "lqg",
    dataset_modes=DATASET_MODES,
    init_seed: int = 123,
    noise_seed: int = 10_000,
    random_signal_kwargs: dict | None = None,
    verbose: bool = True,
    **env_kwargs,
):
    """Generate aligned trajectory datasets for the requested rollout modes."""
    dataset_modes = _validate_inputs(env_id, ctrl_mode, n_traj, dataset_modes)
    resolved_random_signal_kwargs = _resolve_random_signal_kwargs(random_signal_kwargs)

    env_init, envs = _make_envs(env_id, dataset_modes, **env_kwargs)
    controllers = _make_controllers(
        envs,
        dataset_modes,
        ctrl_mode,
        resolved_random_signal_kwargs,
    )

    reference_env = next(iter(envs.values()))
    n_state, n_action, n_steps = (
        reference_env.n_state,
        reference_env.n_action,
        reference_env.n_steps,
    )
    out = _allocate_dataset_arrays(dataset_modes, n_traj, n_state, n_action, n_steps)
    b2 = _extract_shared_b2(*tuple(envs.values()))

    out["traj_times_sec"] = np.zeros(n_traj, dtype=np.float64)
    t_global_start = time.perf_counter()

    for k in range(n_traj):
        t0 = time.perf_counter()

        x0 = _sample_initial_state(env_init, seed=init_seed + k)
        out["init_states"][k] = x0

        seed_k = noise_seed + k
        for mode in dataset_modes:
            rollout_mode = ctrl_mode if mode == "ctrl" else mode
            X, U, R, T = rollout_with_actions(
                envs[mode],
                controllers[mode],
                rollout_mode,
                x0,
                seed_k,
            )
            _store_one_rollout(out, mode, b2, k, X, U, R, T)

        dt = time.perf_counter() - t0
        out["traj_times_sec"][k] = dt

        if verbose:
            elapsed = time.perf_counter() - t_global_start
            mean_time = out["traj_times_sec"][: k + 1].mean()
            remaining = mean_time * (n_traj - k - 1)
            print(
                f"[{k + 1}/{n_traj}] traj={dt:.3f}s | "
                f"elapsed={elapsed:.2f}s | "
                f"eta={remaining:.2f}s"
            )

    total_elapsed = time.perf_counter() - t_global_start

    out["env_id"] = env_id
    out["ctrl_mode"] = ctrl_mode if "ctrl" in dataset_modes else None
    out["dataset_modes"] = dataset_modes
    out["env_kwargs"] = dict(env_kwargs)
    out["random_signal_kwargs"] = dict(resolved_random_signal_kwargs)
    out["seeds"] = {
        "init_seed": init_seed,
        "noise_seed": noise_seed,
    }
    out["B2"] = b2
    out["total_time_sec"] = float(total_elapsed)
    out["mean_time_per_traj_sec"] = float(out["traj_times_sec"].mean())

    if verbose:
        print(
            f"\nCompleted {n_traj} trajectories "
            f"in {total_elapsed:.2f}s "
            f"(mean {out['mean_time_per_traj_sec']:.3f}s / traj)"
        )

    return out


def build_export_dataset(data, mode="ctrl", fill_nan=False):
    """Convert plotting-friendly arrays into time-major export arrays."""
    available_modes = tuple(data.get("dataset_modes", DATASET_MODES))
    if mode not in available_modes:
        raise ValueError(f"mode must be one of {available_modes}.")

    X = data["X"][mode]
    U = data["U"][mode]
    F = data["U_field"][mode]
    seeds = data.get("seeds") or {}

    ds = {
        "solutions": np.transpose(X, (0, 2, 1)),
        "init_states": data["init_states"][:, None, :].copy(),
        "controls": np.transpose(U, (0, 2, 1)),
        "controls_field": np.transpose(F, (0, 2, 1)),
        "R": data["R"][mode].copy(),
        "T": data["T"][mode].copy(),
        "env_id": data.get("env_id"),
        "ctrl_mode": data.get("ctrl_mode"),
        "dataset_modes": available_modes,
        "env_kwargs": dict(data.get("env_kwargs", {})),
        "random_signal_kwargs": dict(data.get("random_signal_kwargs", {})),
        "seeds": dict(seeds),
        "export_mode": mode,
    }

    if fill_nan:
        for key in ("solutions", "controls", "controls_field"):
            ds[key] = np.nan_to_num(ds[key])

    return ds


def sample_trajectory_indices(n_total: int, sample_size: int, seed: int) -> np.ndarray:
    """Sample distinct trajectory indices for console checks."""
    if n_total <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    n_sample = min(sample_size, n_total)
    return np.sort(rng.choice(n_total, size=n_sample, replace=False))


def print_sample_shape_checks(data, idxs: np.ndarray) -> None:
    """Print per-trajectory tensor shapes for a sampled subset."""
    modes = tuple(data.get("dataset_modes", DATASET_MODES))
    print(f"\nShape checks on {len(idxs)} sampled trajectories: {idxs.tolist()}")
    for i in idxs:
        print(f"\nTrajectory {int(i)}")
        for mode in modes:
            print(f"X[{mode}]: {data['X'][mode][i].shape}")
            print(f"U[{mode}]: {data['U'][mode][i].shape}")
            print(f"U_field[{mode}]: {data['U_field'][mode][i].shape}")
            print(f"R[{mode}]: {data['R'][mode][i]}")
            print(f"T[{mode}]: {data['T'][mode][i]}")


def check_dataset_diversity(data, idxs: np.ndarray | None = None, dec: int = 10) -> None:
    """Print initialization consistency and uniqueness checks."""
    total_N = data["init_states"].shape[0]
    if idxs is None:
        idxs = np.arange(total_N)

    idxs = np.asarray(idxs, dtype=int)
    modes = tuple(data.get("dataset_modes", DATASET_MODES))

    same_init_by_mode = {
        mode: np.allclose(data["init_states"][idxs], data["X"][mode][idxs, :, 0])
        for mode in modes
    }

    uniq_init = np.unique(np.round(data["init_states"][idxs], dec), axis=0).shape[0]

    unique_rollouts = {}
    for mode in modes:
        signatures = []
        for k in idxs:
            T = int(data["T"][mode][k])
            Xk = np.round(data["X"][mode][k, :, : T + 1], dec)
            Uk = np.round(data["U"][mode][k, :, : T], dec)
            signatures.append((Xk.tobytes(), Uk.tobytes()))
        unique_rollouts[mode] = len(set(signatures))

    print(
        f"\nDiversity checks on {len(idxs)} sampled trajectories out of {total_N}: "
        f"{idxs.tolist()}"
    )
    for mode in modes:
        print(f"init_states == X[{mode}][:,:,0]: {same_init_by_mode[mode]}")
    print(f"unique initial states: {uniq_init}/{len(idxs)}")
    for mode in modes:
        print(f"unique {mode} rollouts: {unique_rollouts[mode]}/{len(idxs)}")


def _parse_key_value_items(items: list[str]) -> dict:
    """Parse repeated KEY=VALUE CLI items using literal_eval when possible."""
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got: {item}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        parsed[key] = value
    return parsed


def _stringify_key_value_items(items: dict) -> list[str]:
    """Convert a dictionary into KEY=VALUE strings for argparse defaults."""
    return [f"{key}={repr(value)}" for key, value in items.items()]


def _maybe_add(target: dict, key: str, value) -> None:
    """Add a key only when its CLI value is explicitly set."""
    if value is not None:
        target[key] = value


def _to_serializable(value):
    """Convert nested data into JSON-safe Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _load_yaml_config(path: Path | None) -> dict:
    """Load a YAML config file if provided."""
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a top-level mapping.")

    for key in ("env_kwargs", "random_signal_kwargs"):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"'{key}' in YAML config must be a mapping.")

    return config


def _parser_defaults_from_config(config: dict) -> dict:
    """Flatten YAML config values into argparse defaults."""
    defaults = {}

    for key in (
        "env_id",
        "num_traj",
        "ctrl_mode",
        "dataset_modes",
        "init_seed",
        "noise_seed",
        "sample_check_count",
        "sample_check_seed",
        "fill_nan",
    ):
        if key in config:
            defaults[key] = config[key]

    if "output_dir" in config:
        defaults["output_dir"] = Path(config["output_dir"])

    env_kwargs = dict(config.get("env_kwargs", {}))
    extra_env_kwargs = {}
    for key in ENV_ARG_KEYS:
        if key in env_kwargs:
            defaults[key] = env_kwargs.pop(key)
    if env_kwargs:
        extra_env_kwargs.update(env_kwargs)
    if extra_env_kwargs:
        defaults["env_kwarg"] = _stringify_key_value_items(extra_env_kwargs)

    random_signal_kwargs = dict(config.get("random_signal_kwargs", {}))
    extra_random_kwargs = {}
    for key, arg_name in RANDOM_ARG_MAP.items():
        if key in random_signal_kwargs:
            defaults[arg_name] = random_signal_kwargs.pop(key)
    if extra_random_kwargs:
        defaults["random_kwarg"] = _stringify_key_value_items(extra_random_kwargs)
    if random_signal_kwargs:
        defaults["random_kwarg"] = _stringify_key_value_items(random_signal_kwargs)

    return defaults


def _build_env_kwargs(args: argparse.Namespace) -> dict:
    """Collect common PDE kwargs and merge any extra per-env overrides."""
    env_kwargs = {}
    _maybe_add(env_kwargs, "n_steps", args.n_steps)
    _maybe_add(env_kwargs, "domain_length", args.domain_length)
    _maybe_add(env_kwargs, "integration_time", args.integration_time)
    _maybe_add(env_kwargs, "sample_time", args.sample_time)
    _maybe_add(env_kwargs, "process_noise_cov", args.process_noise_cov)
    _maybe_add(env_kwargs, "sensor_noise_cov", args.sensor_noise_cov)
    _maybe_add(env_kwargs, "n_state", args.n_state)
    _maybe_add(env_kwargs, "n_observation", args.n_observation)
    _maybe_add(env_kwargs, "n_action", args.n_action)
    _maybe_add(env_kwargs, "control_sup_width", args.control_sup_width)
    _maybe_add(env_kwargs, "Q_weight", args.Q_weight)
    _maybe_add(env_kwargs, "R_weight", args.R_weight)
    _maybe_add(env_kwargs, "action_limit", args.action_limit)
    _maybe_add(env_kwargs, "observation_limit", args.observation_limit)
    _maybe_add(env_kwargs, "reward_limit", args.reward_limit)
    env_kwargs.update(_parse_key_value_items(args.env_kwarg))
    return env_kwargs


def _build_random_signal_kwargs(args: argparse.Namespace) -> dict:
    """Collect SmoothRandom kwargs from the CLI."""
    random_signal_kwargs = {
        "signal_mode": args.random_signal_mode,
        "freq_range": tuple(args.random_freq_range),
        "min_val": args.random_min_val,
        "max_val": args.random_max_val,
        "amp_range": tuple(args.random_amp_range),
        "num_modes": args.random_num_modes,
        "zero_mean": args.random_zero_mean,
        "per_dim": args.random_per_dim,
    }
    if args.random_cycles_range is not None:
        random_signal_kwargs["cycles_range"] = tuple(args.random_cycles_range)
    random_signal_kwargs.update(_parse_key_value_items(args.random_kwarg))
    return random_signal_kwargs


def print_hyperparameters(
    config: dict,
    env_kwargs: dict,
    random_signal_kwargs: dict,
) -> None:
    """Print the full run configuration before generation starts."""
    payload = {
        "env_id": config["env_id"],
        "num_traj": config["num_traj"],
        "ctrl_mode": config["ctrl_mode"],
        "dataset_modes": config["dataset_modes"],
        "init_seed": config["init_seed"],
        "noise_seed": config["noise_seed"],
        "sample_check_count": config["sample_check_count"],
        "sample_check_seed": config["sample_check_seed"],
        "fill_nan": config["fill_nan"],
        "output_dir": config["output_dir"],
        "env_kwargs": env_kwargs,
        "random_signal_kwargs": random_signal_kwargs,
    }
    print("Run hyperparameters:")
    print(pformat(_to_serializable(payload), sort_dicts=True))


def save_export_datasets(
    data,
    num_traj: int,
    output_dir: Path,
    fill_nan: bool,
) -> list[Path]:
    """Save one compressed NPZ file per requested mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    modes = tuple(data.get("dataset_modes", DATASET_MODES))
    for mode in modes:
        ds = build_export_dataset(data, mode=mode, fill_nan=fill_nan)
        out_path = output_dir / f"{data['env_id']}_dataset_{mode}_{num_traj}.npz"
        np.savez_compressed(
            out_path,
            solutions=ds["solutions"],
            init_states=ds["init_states"],
            controls=ds["controls"],
            controls_field=ds["controls_field"],
            R=ds["R"],
            T=ds["T"],
            B2=data["B2"],
            env_id=np.array(ds["env_id"]),
            ctrl_mode=np.array(ds["ctrl_mode"]),
            export_mode=np.array(ds["export_mode"]),
            dataset_modes_json=np.array(json.dumps(_to_serializable(ds["dataset_modes"]))),
            env_kwargs_json=np.array(json.dumps(_to_serializable(ds["env_kwargs"]), sort_keys=True)),
            random_signal_kwargs_json=np.array(
                json.dumps(_to_serializable(ds["random_signal_kwargs"]), sort_keys=True)
            ),
            seeds_json=np.array(json.dumps(_to_serializable(ds["seeds"]), sort_keys=True)),
            total_time_sec=np.array(data["total_time_sec"]),
            mean_time_per_traj_sec=np.array(data["mean_time_per_traj_sec"]),
        )
        saved_paths.append(out_path)
        print(f"Saved {mode} dataset to {out_path}")

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate aligned zero/random/closed-loop datasets for linear PDE environments."
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML config file. CLI flags override YAML values.",
    )
    parser.add_argument("--env-id", choices=sorted(LINEAR_PDE_IDS))
    parser.add_argument("--num-traj", type=int)
    parser.add_argument("--ctrl-mode", default="lqg", choices=sorted(VALID_CTRL_MODES))
    parser.add_argument(
        "--dataset-modes",
        nargs="+",
        choices=DATASET_MODES,
        help="Subset of rollout modes to generate, e.g. --dataset-modes zero ctrl",
    )
    parser.add_argument("--init-seed", default=123, type=int)
    parser.add_argument("--noise-seed", default=10_000, type=int)
    parser.add_argument("--sample-check-count", default=5, type=int)
    parser.add_argument("--sample-check-seed", default=42, type=int)
    parser.add_argument("--fill-nan", action="store_true")
    parser.add_argument("--output-dir", default=Path("."), type=Path)

    # Common PDE environment kwargs. Leave them optional so env defaults still work.
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--domain-length", type=float)
    parser.add_argument("--integration-time", type=float)
    parser.add_argument("--sample-time", type=float)
    parser.add_argument("--process-noise-cov", type=float)
    parser.add_argument("--sensor-noise-cov", type=float)
    parser.add_argument("--n-state", type=int)
    parser.add_argument("--n-observation", type=int)
    parser.add_argument("--n-action", type=int)
    parser.add_argument("--control-sup-width", type=float)
    parser.add_argument("--Q-weight", type=float)
    parser.add_argument("--R-weight", type=float)
    parser.add_argument("--action-limit", type=float)
    parser.add_argument("--observation-limit", type=float)
    parser.add_argument("--reward-limit", type=float)
    parser.add_argument(
        "--env-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment kwargs, parsed with ast.literal_eval when possible.",
    )

    # SmoothRandom kwargs.
    parser.add_argument("--random-signal-mode", default="manual")
    parser.add_argument("--random-freq-range", nargs=2, type=float, default=(1.0, 3.0))
    parser.add_argument("--random-min-val", type=float, default=-1.0)
    parser.add_argument("--random-max-val", type=float, default=1.0)
    parser.add_argument("--random-amp-range", nargs=2, type=float, default=(10.0, 20.0))
    parser.add_argument("--random-cycles-range", nargs=2, type=float)
    parser.add_argument("--random-num-modes", type=int, default=4)
    parser.add_argument("--random-zero-mean", dest="random_zero_mean", action="store_true")
    parser.add_argument("--no-random-zero-mean", dest="random_zero_mean", action="store_false")
    parser.set_defaults(random_zero_mean=True)
    parser.add_argument("--random-per-dim", dest="random_per_dim", action="store_true")
    parser.add_argument("--no-random-per-dim", dest="random_per_dim", action="store_false")
    parser.set_defaults(random_per_dim=True)
    parser.add_argument(
        "--random-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra SmoothRandom kwargs, parsed with ast.literal_eval when possible.",
    )

    return parser


def _merge_cli_and_yaml_config(
    parser: argparse.ArgumentParser,
    cli_args: argparse.Namespace,
) -> dict:
    """Merge base defaults, YAML config, and CLI values into one runtime config."""
    config = json.loads(json.dumps(DEFAULT_RUN_CONFIG))
    yaml_config = _load_yaml_config(getattr(cli_args, "config", None))
    config.update({k: v for k, v in yaml_config.items() if k not in {"env_kwargs", "random_signal_kwargs"}})

    config["env_kwargs"] = dict(DEFAULT_RUN_CONFIG["env_kwargs"])
    config["env_kwargs"].update(yaml_config.get("env_kwargs", {}))

    config["random_signal_kwargs"] = dict(DEFAULT_RUN_CONFIG["random_signal_kwargs"])
    config["random_signal_kwargs"].update(yaml_config.get("random_signal_kwargs", {}))

    cli_overrides = vars(cli_args).copy()
    cli_overrides.pop("config", None)

    for key in (
        "env_id",
        "num_traj",
        "ctrl_mode",
        "dataset_modes",
        "init_seed",
        "noise_seed",
        "sample_check_count",
        "sample_check_seed",
        "fill_nan",
        "output_dir",
    ):
        if key in cli_overrides and cli_overrides[key] is not None:
            config[key] = cli_overrides[key]

    for key in ENV_ARG_KEYS:
        if key in cli_overrides and cli_overrides[key] is not None:
            config["env_kwargs"][key] = cli_overrides[key]
    if cli_overrides.get("env_kwarg"):
        config["env_kwargs"].update(_parse_key_value_items(cli_overrides["env_kwarg"]))

    for random_key, arg_name in RANDOM_ARG_MAP.items():
        if arg_name in cli_overrides and cli_overrides[arg_name] is not None:
            value = cli_overrides[arg_name]
            if random_key in {"freq_range", "amp_range", "cycles_range"} and value is not None:
                value = list(value)
            config["random_signal_kwargs"][random_key] = value
    if cli_overrides.get("random_kwarg"):
        config["random_signal_kwargs"].update(_parse_key_value_items(cli_overrides["random_kwarg"]))

    if "output_dir" in config:
        config["output_dir"] = Path(config["output_dir"])

    if "env_id" not in config or config["env_id"] is None:
        parser.error("Provide --env-id on the CLI or in the YAML config.")
    if "num_traj" not in config or config["num_traj"] is None:
        parser.error("Provide --num-traj on the CLI or in the YAML config.")

    return config


def main() -> None:
    """Run the end-to-end CLI workflow."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()

    parser = build_parser()
    parser.set_defaults(**_parser_defaults_from_config(_load_yaml_config(config_args.config)))
    args = parser.parse_args()

    config = _merge_cli_and_yaml_config(parser, args)
    env_kwargs = dict(config["env_kwargs"])
    random_signal_kwargs = _resolve_random_signal_kwargs(config["random_signal_kwargs"])
    print_hyperparameters(config, env_kwargs, random_signal_kwargs)

    # Step 1: generate the full aligned dataset with per-trajectory timing prints.
    data = generate_controlled_dataset(
        env_id=config["env_id"],
        n_traj=config["num_traj"],
        ctrl_mode=config["ctrl_mode"],
        dataset_modes=tuple(config["dataset_modes"]),
        init_seed=config["init_seed"],
        noise_seed=config["noise_seed"],
        random_signal_kwargs=random_signal_kwargs,
        verbose=True,
        **env_kwargs,
    )

    # Step 2: print shape checks and diversity on 5 sampled trajectories.
    idxs = sample_trajectory_indices(
        n_total=data["init_states"].shape[0],
        sample_size=config["sample_check_count"],
        seed=config["sample_check_seed"],
    )
    print_sample_shape_checks(data, idxs)
    check_dataset_diversity(data, idxs=idxs)

    # Step 3: export the requested modes to compressed NPZ files.
    save_export_datasets(
        data=data,
        num_traj=config["num_traj"],
        output_dir=config["output_dir"],
        fill_nan=config["fill_nan"],
    )


if __name__ == "__main__":
    main()
