#!/usr/bin/env python3
"""Command-line dataset generation for nonlinear PDE environments.

Run directly from the repository root:

1. With the default YAML config:
   `.venv/bin/python nonlinearPDE_data_gen.py --config nonlinearPDE_data_gen.default.yaml`

2. With YAML config plus CLI overrides:
   `.venv/bin/python nonlinearPDE_data_gen.py --config nonlinearPDE_data_gen.default.yaml --num-traj 300 --dataset-modes zero random`

3. Without YAML, passing key arguments on the command line:
   `.venv/bin/python nonlinearPDE_data_gen.py --env-id kuramoto_sivashinsky --num-traj 300 --dataset-modes zero --n-state 256 --n-observation 256 --n-action 8 --n-steps 400 --process-noise-cov 0 --sensor-noise-cov 1e-8`

4. Run with PPO ctrl from a checkpoint directory:
   `.venv/bin/python nonlinearPDE_data_gen.py --config nonlinearPDE_data_gen.default.yaml --dataset-modes zero ctrl --ppo-checkpoint-dir /path/to/ppo_run`

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
import torch
import yaml

import controlgym as gym


NONLINEAR_PDE_IDS = {
    "allen_cahn",
    "burgers",
    "cahn_hilliard",
    "fisher",
    "ginzburg_landau",
    "korteweg_de_vries",
    "kuramoto_sivashinsky",
}
VALID_NL_CTRL_MODES = {"ppo"}
DATASET_MODES = ("zero", "random", "ctrl")
DEFAULT_DATASET_MODES = ("zero", "random")
DEFAULT_RANDOM_SIGNAL_KWARGS = {
    "signal_mode": "manual",
    "freq_range": (1, 3),
    "min_val": -1,
    "max_val": 1,
    "amp_range": (10, 20),
}
PPO_INIT_KEYS = {"actor_hidden_dim", "critic_hidden_dim", "lr", "discount_factor", "device"}
PPO_TRAIN_KEYS = {
    "num_train_iter",
    "num_episodes_per_iter",
    "episode_length",
    "sgd_epoch_num",
    "mini_batch_size",
    "clip",
    "cov_param",
}
DEFAULT_RUN_CONFIG = {
    "ctrl_mode": "ppo",
    "dataset_modes": ["zero"],
    "init_seed": 123,
    "noise_seed": 10_000,
    "sample_check_count": 5,
    "sample_check_seed": 42,
    "fill_nan": False,
    "output_dir": ".",
    "train_controller": False,
    "ppo_checkpoint_dir": None,
    "env_kwargs": {
        "n_state": 256,
        "n_observation": 256,
        "n_action": 8,
        "n_steps": 400,
        "process_noise_cov": 0.0,
        "sensor_noise_cov": 1e-8,
        "action_limit": 1.0,
    },
    "random_signal_kwargs": {
        "signal_mode": "manual",
        "freq_range": [1, 3],
        "min_val": -1,
        "max_val": 1,
        "amp_range": [10, 20],
    },
    "ppo_train_kwargs": {
        "actor_hidden_dim": 64,
        "critic_hidden_dim": 64,
        "lr": 1e-4,
        "discount_factor": 0.99,
        "device": "cpu",
        "num_train_iter": 5,
        "num_episodes_per_iter": 8,
        "episode_length": 100,
        "sgd_epoch_num": 2,
        "mini_batch_size": 8,
        "clip": 0.2,
        "cov_param": 0.05,
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
}

PPO_ARG_MAP = {
    "actor_hidden_dim": "ppo_actor_hidden_dim",
    "critic_hidden_dim": "ppo_critic_hidden_dim",
    "lr": "ppo_lr",
    "discount_factor": "ppo_discount_factor",
    "device": "ppo_device",
    "num_train_iter": "ppo_num_train_iter",
    "num_episodes_per_iter": "ppo_num_episodes_per_iter",
    "episode_length": "ppo_episode_length",
    "sgd_epoch_num": "ppo_sgd_epoch_num",
    "mini_batch_size": "ppo_mini_batch_size",
    "clip": "ppo_clip",
    "cov_param": "ppo_cov_param",
}


def _validate_dataset_modes(dataset_modes) -> tuple[str, ...]:
    dataset_modes = tuple(dataset_modes)
    if len(dataset_modes) == 0:
        raise ValueError("dataset_modes must contain at least one mode.")
    if len(set(dataset_modes)) != len(dataset_modes):
        raise ValueError("dataset_modes contains duplicate modes.")
    unknown = set(dataset_modes) - set(DATASET_MODES)
    if unknown:
        raise ValueError(f"Unknown dataset modes: {unknown}. Allowed: {DATASET_MODES}")
    return dataset_modes


def _validate_nonlinear_inputs(
    env_id: str,
    n_traj: int,
    dataset_modes,
    ctrl_mode: str,
    controller,
    train_controller: bool,
):
    if env_id not in NONLINEAR_PDE_IDS:
        raise ValueError(f"env_id must be one of {NONLINEAR_PDE_IDS}. Got: {env_id}")
    if n_traj <= 0:
        raise ValueError("n_traj must be a positive integer.")

    dataset_modes = _validate_dataset_modes(dataset_modes)
    if "ctrl" in dataset_modes:
        if ctrl_mode not in VALID_NL_CTRL_MODES:
            raise ValueError(f"ctrl_mode must be one of {VALID_NL_CTRL_MODES}. Got: {ctrl_mode}")
        if controller is None and not train_controller:
            raise ValueError(
                "'ctrl' in dataset_modes requires a provided PPO controller, "
                "or set train_controller=True to train internally."
            )

    return dataset_modes


def _make_nonlinear_envs(env_id: str, dataset_modes, **env_kwargs):
    env_init = gym.make(env_id, **env_kwargs)
    envs = {mode: gym.make(env_id, **env_kwargs) for mode in dataset_modes}
    return env_init, envs


def _sample_initial_state(env_init, seed: int) -> np.ndarray:
    _, info = env_init.reset(seed=seed)
    return info["state"].copy()


def _set_zero_target_if_supported(env) -> None:
    if hasattr(env, "set_target_state"):
        env.set_target_state(np.zeros(env.n_state))


def _resolve_random_signal_kwargs(random_signal_kwargs: dict | None) -> dict:
    resolved = dict(DEFAULT_RANDOM_SIGNAL_KWARGS)
    if random_signal_kwargs is not None:
        resolved.update(dict(random_signal_kwargs))
    resolved.pop("env", None)
    resolved.pop("seed", None)
    return resolved


def _split_ppo_kwargs(ppo_train_kwargs: dict | None):
    ppo_train_kwargs = {} if ppo_train_kwargs is None else dict(ppo_train_kwargs)
    init_kwargs = {k: ppo_train_kwargs[k] for k in PPO_INIT_KEYS if k in ppo_train_kwargs}
    train_kwargs = {k: ppo_train_kwargs[k] for k in PPO_TRAIN_KEYS if k in ppo_train_kwargs}
    return init_kwargs, train_kwargs


def _build_or_train_ppo_controller(
    env_ctrl,
    controller,
    train_controller: bool,
    ppo_train_kwargs: dict | None,
):
    init_kwargs, train_kwargs = _split_ppo_kwargs(ppo_train_kwargs)

    if controller is None:
        controller = gym.controllers.PPO(env_ctrl, **init_kwargs)
        controller_source = "new"
    else:
        if getattr(controller.env, "n_observation", None) != env_ctrl.n_observation:
            raise ValueError("Provided PPO controller observation dimension does not match env_ctrl.")
        if getattr(controller.env, "n_action", None) != env_ctrl.n_action:
            raise ValueError("Provided PPO controller action dimension does not match env_ctrl.")
        controller.env = env_ctrl
        controller_source = "provided"

    trained_in_function = False
    if train_controller:
        controller.train(**train_kwargs)
        trained_in_function = True

    return controller, init_kwargs, train_kwargs, controller_source, trained_in_function


def _build_controllers(
    envs,
    dataset_modes,
    controller,
    ctrl_mode: str,
    train_controller: bool,
    ppo_train_kwargs: dict | None,
    random_signal_kwargs: dict,
):
    controllers = {}
    ppo_init_kwargs = {}
    ppo_train_run_kwargs = {}
    controller_source = "not_requested"
    trained_in_function = False

    if "zero" in dataset_modes:
        controllers["zero"] = gym.controllers.Zero(envs["zero"])

    if "random" in dataset_modes:
        controllers["random"] = gym.controllers.SmoothRandom(envs["random"], **random_signal_kwargs)

    if "ctrl" in dataset_modes:
        if ctrl_mode != "ppo":
            raise ValueError("Only ctrl_mode='ppo' is supported for nonlinear PDEs.")
        (
            controllers["ctrl"],
            ppo_init_kwargs,
            ppo_train_run_kwargs,
            controller_source,
            trained_in_function,
        ) = _build_or_train_ppo_controller(
            envs["ctrl"],
            controller=controller,
            train_controller=train_controller,
            ppo_train_kwargs=ppo_train_kwargs,
        )

    return controllers, ppo_init_kwargs, ppo_train_run_kwargs, controller_source, trained_in_function


def _select_action_nonlinear(mode: str, controller, observation: np.ndarray, n_action: int) -> np.ndarray:
    if mode == "zero":
        return np.zeros(n_action)
    if mode == "random":
        return controller.select_action()
    if mode == "ppo":
        return controller.select_action(observation, cov_param=0.0)
    raise ValueError(f"Unknown nonlinear rollout mode: {mode}")


def _rollout_with_actions_nonlinear(
    env,
    controller,
    rollout_mode: str,
    x0: np.ndarray,
    noise_seed: int,
):
    if rollout_mode == "random":
        controller.reset(seed=noise_seed)

    _set_zero_target_if_supported(env)
    observation, info = env.reset(seed=noise_seed, state=x0)

    X = np.full((env.n_state, env.n_steps + 1), np.nan)
    U = np.full((env.n_action, env.n_steps), np.nan)
    X[:, 0] = info["state"]

    total_reward = 0.0

    for t in range(env.n_steps):
        action = _select_action_nonlinear(rollout_mode, controller, observation, env.n_action)
        observation, reward, terminated, truncated, info = env.step(action)

        U[:, t] = action
        X[:, t + 1] = info["state"]
        total_reward += reward

        if terminated or truncated:
            T = t + 1
            return X[:, : T + 1], U[:, : T], total_reward, T

    return X, U, total_reward, env.n_steps


def _get_control_lifting_map(env):
    if hasattr(env, "B2") and getattr(env, "B2") is not None:
        return np.asarray(env.B2).copy(), "B2"
    if hasattr(env, "control_sup") and getattr(env, "control_sup") is not None:
        return np.asarray(env.control_sup).copy(), "control_sup"
    return None, "none"


def _compute_control_field_from_map(lifting_map: np.ndarray | None, U: np.ndarray, n_state: int):
    if lifting_map is None:
        return np.full((n_state, U.shape[1]), np.nan)
    return lifting_map @ U


def _allocate_dataset_arrays(dataset_modes, n_traj: int, n_state: int, n_action: int, n_steps: int):
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


def _store_one_rollout(data, mode: str, lifting_map, n_state: int, k: int, X, U, R: float, T: int):
    data["X"][mode][k, :, : X.shape[1]] = X
    data["U"][mode][k, :, : U.shape[1]] = U
    F = _compute_control_field_from_map(lifting_map, U, n_state)
    data["U_field"][mode][k, :, : F.shape[1]] = F
    data["R"][mode][k] = R
    data["T"][mode][k] = T


def generate_controlled_dataset_nonlinear(
    env_id: str,
    n_traj: int,
    dataset_modes=("zero", "random"),
    controller=None,
    ctrl_mode: str = "ppo",
    train_controller: bool = False,
    ppo_train_kwargs: dict | None = None,
    random_signal_kwargs: dict | None = None,
    init_seed: int = 123,
    noise_seed: int = 10_000,
    verbose: bool = False,
    **env_kwargs,
):
    dataset_modes = _validate_nonlinear_inputs(
        env_id=env_id,
        n_traj=n_traj,
        dataset_modes=dataset_modes,
        ctrl_mode=ctrl_mode,
        controller=controller,
        train_controller=train_controller,
    )
    resolved_random_signal_kwargs = _resolve_random_signal_kwargs(random_signal_kwargs)

    env_init, envs = _make_nonlinear_envs(env_id, dataset_modes, **env_kwargs)
    for env in envs.values():
        _set_zero_target_if_supported(env)

    (
        controllers,
        ppo_init_kwargs,
        ppo_train_run_kwargs,
        controller_source,
        trained_in_function,
    ) = _build_controllers(
        envs,
        dataset_modes,
        controller=controller,
        ctrl_mode=ctrl_mode,
        train_controller=train_controller,
        ppo_train_kwargs=ppo_train_kwargs,
        random_signal_kwargs=resolved_random_signal_kwargs,
    )

    reference_env = next(iter(envs.values()))
    n_state, n_action, n_steps = reference_env.n_state, reference_env.n_action, reference_env.n_steps
    out = _allocate_dataset_arrays(dataset_modes, n_traj, n_state, n_action, n_steps)

    lifting_maps = {}
    lifting_types = {}
    for mode in dataset_modes:
        lifting_maps[mode], lifting_types[mode] = _get_control_lifting_map(envs[mode])

    out["traj_times_sec"] = np.zeros(n_traj, dtype=np.float64)
    t_global_start = time.perf_counter()

    for k in range(n_traj):
        t0 = time.perf_counter()

        x0 = _sample_initial_state(env_init, seed=init_seed + k)
        out["init_states"][k] = x0

        seed_k = noise_seed + k
        for mode in dataset_modes:
            rollout_mode = "ppo" if mode == "ctrl" else mode
            X, U, R, T = _rollout_with_actions_nonlinear(
                envs[mode],
                controller=controllers[mode],
                rollout_mode=rollout_mode,
                x0=x0,
                noise_seed=seed_k,
            )
            _store_one_rollout(out, mode, lifting_maps[mode], n_state, k, X, U, R, T)

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
    out["controller_class"] = None if "ctrl" not in dataset_modes else controllers["ctrl"].__class__.__name__
    out["controller_source"] = controller_source
    out["trained_in_function"] = trained_in_function
    out["ppo_init_kwargs"] = ppo_init_kwargs
    out["ppo_train_kwargs"] = ppo_train_run_kwargs
    out["control_lifting_map_type"] = dict(lifting_types)
    out["control_lifting_map"] = {
        mode: None if lifting_maps[mode] is None else lifting_maps[mode].copy()
        for mode in dataset_modes
    }
    out["seeds"] = {
        "init_seed": init_seed,
        "noise_seed": noise_seed,
    }
    out["total_time_sec"] = float(total_elapsed)
    out["mean_time_per_traj_sec"] = float(out["traj_times_sec"].mean())

    if verbose:
        print(
            f"\nCompleted {n_traj} trajectories "
            f"in {total_elapsed:.2f}s "
            f"(mean {out['mean_time_per_traj_sec']:.3f}s / traj)"
        )

    return out


def build_export_dataset(data, mode="zero", fill_nan=False):
    """Convert plotting-friendly arrays into time-major export arrays."""
    available_modes = tuple(data.get("dataset_modes", DATASET_MODES))
    if mode not in available_modes:
        raise ValueError(f"mode must be one of {available_modes}.")

    X = data["X"][mode]
    U = data["U"][mode]
    F = data["U_field"][mode]
    seeds = data.get("seeds") or {}
    lifting_map_types = data.get("control_lifting_map_type") or {}
    lifting_maps = data.get("control_lifting_map") or {}

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
        "control_lifting_map_type": lifting_map_types.get(mode, "none"),
        "control_lifting_map": None if lifting_maps.get(mode) is None else lifting_maps[mode].copy(),
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
    total_n = data["init_states"].shape[0]
    if idxs is None:
        idxs = np.arange(total_n)

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
        f"\nDiversity checks on {len(idxs)} sampled trajectories out of {total_n}: "
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
    if isinstance(value, torch.device):
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

    for key in ("env_kwargs", "random_signal_kwargs", "ppo_train_kwargs"):
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
        "train_controller",
    ):
        if key in config:
            defaults[key] = config[key]

    if "output_dir" in config:
        defaults["output_dir"] = Path(config["output_dir"])
    if config.get("ppo_checkpoint_dir") is not None:
        defaults["ppo_checkpoint_dir"] = Path(config["ppo_checkpoint_dir"])

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

    ppo_train_kwargs = dict(config.get("ppo_train_kwargs", {}))
    extra_ppo_kwargs = {}
    for key, arg_name in PPO_ARG_MAP.items():
        if key in ppo_train_kwargs:
            defaults[arg_name] = ppo_train_kwargs.pop(key)
    if extra_ppo_kwargs:
        defaults["ppo_kwarg"] = _stringify_key_value_items(extra_ppo_kwargs)
    if ppo_train_kwargs:
        defaults["ppo_kwarg"] = _stringify_key_value_items(ppo_train_kwargs)

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
    }
    if args.random_cycles_range is not None:
        random_signal_kwargs["cycles_range"] = tuple(args.random_cycles_range)
    random_signal_kwargs.update(_parse_key_value_items(args.random_kwarg))
    return random_signal_kwargs


def _build_ppo_train_kwargs(args: argparse.Namespace) -> dict:
    """Collect PPO init/train kwargs from the CLI."""
    ppo_train_kwargs = {}
    _maybe_add(ppo_train_kwargs, "actor_hidden_dim", args.ppo_actor_hidden_dim)
    _maybe_add(ppo_train_kwargs, "critic_hidden_dim", args.ppo_critic_hidden_dim)
    _maybe_add(ppo_train_kwargs, "lr", args.ppo_lr)
    _maybe_add(ppo_train_kwargs, "discount_factor", args.ppo_discount_factor)
    _maybe_add(ppo_train_kwargs, "device", args.ppo_device)
    _maybe_add(ppo_train_kwargs, "num_train_iter", args.ppo_num_train_iter)
    _maybe_add(ppo_train_kwargs, "num_episodes_per_iter", args.ppo_num_episodes_per_iter)
    _maybe_add(ppo_train_kwargs, "episode_length", args.ppo_episode_length)
    _maybe_add(ppo_train_kwargs, "sgd_epoch_num", args.ppo_sgd_epoch_num)
    _maybe_add(ppo_train_kwargs, "mini_batch_size", args.ppo_mini_batch_size)
    _maybe_add(ppo_train_kwargs, "clip", args.ppo_clip)
    _maybe_add(ppo_train_kwargs, "cov_param", args.ppo_cov_param)
    ppo_train_kwargs.update(_parse_key_value_items(args.ppo_kwarg))
    return ppo_train_kwargs


def _normalize_ppo_train_kwargs(ppo_train_kwargs: dict | None) -> dict:
    """Convert YAML/CLI PPO values into controller-ready kwargs."""
    normalized = {} if ppo_train_kwargs is None else dict(ppo_train_kwargs)
    if "device" in normalized and normalized["device"] is not None:
        device = normalized["device"]
        if not isinstance(device, torch.device):
            normalized["device"] = torch.device(str(device))
    return normalized


def _validate_script_ctrl_config(config: dict) -> None:
    """Raise a script-level error when ctrl was requested without a usable PPO source."""
    dataset_modes = tuple(config["dataset_modes"])
    if "ctrl" in dataset_modes:
        checkpoint_dir = config.get("ppo_checkpoint_dir")
        if checkpoint_dir is None and not config["train_controller"]:
            raise ValueError(
                "'ctrl' in dataset_modes requires train_controller=True or a ppo_checkpoint_dir."
            )


def _build_checkpoint_controller_if_needed(
    env_id: str,
    dataset_modes,
    env_kwargs: dict,
    ppo_train_kwargs: dict,
    ppo_checkpoint_dir: Path | None,
):
    """Instantiate and load a PPO controller when a checkpoint directory is provided."""
    if "ctrl" not in tuple(dataset_modes) or ppo_checkpoint_dir is None:
        return None

    env_ctrl = gym.make(env_id, **env_kwargs)
    _set_zero_target_if_supported(env_ctrl)
    init_kwargs, _ = _split_ppo_kwargs(ppo_train_kwargs)
    controller = gym.controllers.PPO(env_ctrl, **init_kwargs)
    controller.load(test_dir=str(ppo_checkpoint_dir))
    return controller


def print_hyperparameters(
    config: dict,
    env_kwargs: dict,
    random_signal_kwargs: dict,
    ppo_train_kwargs: dict,
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
        "train_controller": config["train_controller"],
        "ppo_checkpoint_dir": config["ppo_checkpoint_dir"],
        "env_kwargs": env_kwargs,
        "random_signal_kwargs": random_signal_kwargs,
        "ppo_train_kwargs": ppo_train_kwargs,
    }
    print("Run hyperparameters:")
    print(pformat(_to_serializable(payload), sort_dicts=True))


def _npz_optional_lifting_map(lifting_map):
    """Return a NumPy value suitable for np.savez_compressed."""
    if lifting_map is None:
        return np.array(None, dtype=object)
    return lifting_map


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
            # Keep the legacy `B2` export key, but save the actual lifting matrix
            # used to map low-dimensional controls onto the full state grid.
            B2=_npz_optional_lifting_map(ds["control_lifting_map"]),
            env_id=np.array(ds["env_id"]),
            ctrl_mode=np.array(ds["ctrl_mode"]),
            export_mode=np.array(ds["export_mode"]),
            dataset_modes_json=np.array(json.dumps(_to_serializable(ds["dataset_modes"]))),
            env_kwargs_json=np.array(json.dumps(_to_serializable(ds["env_kwargs"]), sort_keys=True)),
            random_signal_kwargs_json=np.array(
                json.dumps(_to_serializable(ds["random_signal_kwargs"]), sort_keys=True)
            ),
            seeds_json=np.array(json.dumps(_to_serializable(ds["seeds"]), sort_keys=True)),
            control_lifting_map_type=np.array(ds["control_lifting_map_type"]),
            control_lifting_map=_npz_optional_lifting_map(ds["control_lifting_map"]),
            total_time_sec=np.array(data["total_time_sec"]),
            mean_time_per_traj_sec=np.array(data["mean_time_per_traj_sec"]),
        )
        saved_paths.append(out_path)
        print(f"Saved {mode} dataset to {out_path}")

    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate aligned zero/random/optional-PPO datasets for nonlinear PDE environments."
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Optional YAML config file. CLI flags override YAML values.",
    )
    parser.add_argument("--env-id", choices=sorted(NONLINEAR_PDE_IDS))
    parser.add_argument("--num-traj", type=int)
    parser.add_argument("--ctrl-mode", default="ppo", choices=sorted(VALID_NL_CTRL_MODES))
    parser.add_argument(
        "--dataset-modes",
        nargs="+",
        choices=DATASET_MODES,
        help="Subset of rollout modes to generate, e.g. --dataset-modes zero random ctrl",
    )
    parser.add_argument("--init-seed", default=123, type=int)
    parser.add_argument("--noise-seed", default=10_000, type=int)
    parser.add_argument("--sample-check-count", default=5, type=int)
    parser.add_argument("--sample-check-seed", default=42, type=int)
    parser.add_argument("--fill-nan", action="store_true")
    parser.add_argument("--output-dir", default=Path("."), type=Path)
    parser.add_argument("--train-controller", dest="train_controller", action="store_true")
    parser.add_argument("--no-train-controller", dest="train_controller", action="store_false")
    parser.set_defaults(train_controller=False)
    parser.add_argument(
        "--ppo-checkpoint-dir",
        type=Path,
        help="Directory containing PPO weights saved as ppo_params.pt.",
    )

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

    parser.add_argument("--random-signal-mode", default="manual")
    parser.add_argument("--random-freq-range", nargs=2, type=float, default=(1.0, 3.0))
    parser.add_argument("--random-min-val", type=float, default=-1.0)
    parser.add_argument("--random-max-val", type=float, default=1.0)
    parser.add_argument("--random-amp-range", nargs=2, type=float, default=(10.0, 20.0))
    parser.add_argument("--random-cycles-range", nargs=2, type=float)
    parser.add_argument(
        "--random-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra SmoothRandom kwargs, parsed with ast.literal_eval when possible.",
    )

    parser.add_argument("--ppo-actor-hidden-dim", type=int)
    parser.add_argument("--ppo-critic-hidden-dim", type=int)
    parser.add_argument("--ppo-lr", type=float)
    parser.add_argument("--ppo-discount-factor", type=float)
    parser.add_argument("--ppo-device")
    parser.add_argument("--ppo-num-train-iter", type=int)
    parser.add_argument("--ppo-num-episodes-per-iter", type=int)
    parser.add_argument("--ppo-episode-length", type=int)
    parser.add_argument("--ppo-sgd-epoch-num", type=int)
    parser.add_argument("--ppo-mini-batch-size", type=int)
    parser.add_argument("--ppo-clip", type=float)
    parser.add_argument("--ppo-cov-param", type=float)
    parser.add_argument(
        "--ppo-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra PPO kwargs, parsed with ast.literal_eval when possible.",
    )

    return parser


def _merge_cli_and_yaml_config(
    parser: argparse.ArgumentParser,
    cli_args: argparse.Namespace,
) -> dict:
    """Merge base defaults, YAML config, and CLI values into one runtime config."""
    config = json.loads(json.dumps(DEFAULT_RUN_CONFIG))
    yaml_config = _load_yaml_config(getattr(cli_args, "config", None))
    config.update(
        {
            k: v
            for k, v in yaml_config.items()
            if k not in {"env_kwargs", "random_signal_kwargs", "ppo_train_kwargs"}
        }
    )

    config["env_kwargs"] = dict(DEFAULT_RUN_CONFIG["env_kwargs"])
    config["env_kwargs"].update(yaml_config.get("env_kwargs", {}))

    config["random_signal_kwargs"] = dict(DEFAULT_RUN_CONFIG["random_signal_kwargs"])
    config["random_signal_kwargs"].update(yaml_config.get("random_signal_kwargs", {}))

    config["ppo_train_kwargs"] = dict(DEFAULT_RUN_CONFIG["ppo_train_kwargs"])
    config["ppo_train_kwargs"].update(yaml_config.get("ppo_train_kwargs", {}))

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
        "train_controller",
        "ppo_checkpoint_dir",
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

    for ppo_key, arg_name in PPO_ARG_MAP.items():
        if arg_name in cli_overrides and cli_overrides[arg_name] is not None:
            config["ppo_train_kwargs"][ppo_key] = cli_overrides[arg_name]
    if cli_overrides.get("ppo_kwarg"):
        config["ppo_train_kwargs"].update(_parse_key_value_items(cli_overrides["ppo_kwarg"]))

    if "output_dir" in config:
        config["output_dir"] = Path(config["output_dir"])
    if config.get("ppo_checkpoint_dir") is not None:
        config["ppo_checkpoint_dir"] = Path(config["ppo_checkpoint_dir"])

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
    _validate_script_ctrl_config(config)

    env_kwargs = dict(config["env_kwargs"])
    random_signal_kwargs = _resolve_random_signal_kwargs(config["random_signal_kwargs"])
    ppo_train_kwargs_raw = dict(config["ppo_train_kwargs"])
    ppo_train_kwargs = _normalize_ppo_train_kwargs(ppo_train_kwargs_raw)

    print_hyperparameters(config, env_kwargs, random_signal_kwargs, ppo_train_kwargs_raw)

    controller = _build_checkpoint_controller_if_needed(
        env_id=config["env_id"],
        dataset_modes=tuple(config["dataset_modes"]),
        env_kwargs=env_kwargs,
        ppo_train_kwargs=ppo_train_kwargs,
        ppo_checkpoint_dir=config.get("ppo_checkpoint_dir"),
    )

    data = generate_controlled_dataset_nonlinear(
        env_id=config["env_id"],
        n_traj=config["num_traj"],
        dataset_modes=tuple(config["dataset_modes"]),
        controller=controller,
        ctrl_mode=config["ctrl_mode"],
        train_controller=config["train_controller"],
        ppo_train_kwargs=ppo_train_kwargs,
        random_signal_kwargs=random_signal_kwargs,
        init_seed=config["init_seed"],
        noise_seed=config["noise_seed"],
        verbose=True,
        **env_kwargs,
    )

    idxs = sample_trajectory_indices(
        n_total=data["init_states"].shape[0],
        sample_size=config["sample_check_count"],
        seed=config["sample_check_seed"],
    )
    print_sample_shape_checks(data, idxs)
    check_dataset_diversity(data, idxs=idxs)

    save_export_datasets(
        data=data,
        num_traj=config["num_traj"],
        output_dir=config["output_dir"],
        fill_nan=config["fill_nan"],
    )


if __name__ == "__main__":
    main()
