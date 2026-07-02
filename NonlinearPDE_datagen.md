# Nonlinear PDE Dataset Pipeline

## Scope
Supported nonlinear PDE environments:
- `allen_cahn`
- `burgers`
- `cahn_hilliard`
- `fisher`
- `ginzburg_landau`
- `korteweg_de_vries`
- `kuramoto_sivashinsky`

Closed-loop controller choices:
- `ppo`

Dataset rollout modes:
- `zero`
- `random`
- `ctrl`

`ctrl` is the generic dataset mode name for the closed-loop controller. For nonlinear PDEs, the actual controller type is PPO and is stored in `ctrl_mode`.

## Generator Overview
`generate_controlled_dataset_nonlinear(...)` creates aligned trajectories for every sampled initial state and every requested dataset mode:
- `zero`: zero-valued actions
- `random`: `gym.controllers.SmoothRandom`
- `ctrl`: `gym.controllers.PPO`

For trajectory index `k`:
- initial state seed is `init_seed + k`
- rollout seed is `noise_seed + k`
- all requested modes start from the same `x0`
- the `random` controller regenerates its open-loop action sequence with `random_controller.reset(seed=noise_seed + k)`
- the `ctrl` rollout uses the PPO policy deterministically with `cov_param=0.0`

This guarantees, for the requested `dataset_modes`:
- `N` sampled initial states
- `N` rollouts per requested mode
- the same trajectory index `k` is aligned across all requested modes

The default function-level dataset modes are:
- `zero`
- `random`

The CLI base config starts from `dataset_modes: ["zero"]`, while `nonlinearPDE_data_gen.default.yaml` currently overrides that to `dataset_modes: [zero, random]`.

To generate PPO rollouts, include `ctrl` in `dataset_modes`. A PPO controller must either be provided directly, loaded from a checkpoint through the CLI, or trained inside the generator with `train_controller=True`.

## Dataset Structure
The returned dictionary stores plotting-friendly arrays. The mode keys are exactly the requested `dataset_modes`:

```python
data = {
    "init_states": (N, n_state),
    "X": {
        "zero":   (N, n_state, n_steps + 1),
        "random": (N, n_state, n_steps + 1),
        "ctrl":   (N, n_state, n_steps + 1),
    },
    "U": {
        "zero":   (N, n_action, n_steps),
        "random": (N, n_action, n_steps),
        "ctrl":   (N, n_action, n_steps),
    },
    "U_field": {
        "zero":   (N, n_state, n_steps),
        "random": (N, n_state, n_steps),
        "ctrl":   (N, n_state, n_steps),
    },
    "R": {
        "zero":   (N,),
        "random": (N,),
        "ctrl":   (N,),
    },
    "T": {
        "zero":   (N,),
        "random": (N,),
        "ctrl":   (N,),
    },
}
```

Meanings of the main trajectory statistics:
- `R[mode][k]`: cumulative reward for trajectory `k` in that mode. In these environments, reward is the negative quadratic control cost, so a more negative value means a higher accumulated cost.
- `T[mode][k]`: valid rollout horizon for trajectory `k`. It is the number of executed control steps before termination/truncation, and is used to crop away any NaN padding in `X`, `U`, and `U_field`.
- `control_lifting_map`: actuator-to-state input map for each requested mode. The generator uses `env.B2` when present, then falls back to `env.control_sup`, then stores NaN control fields if neither map exists. The stored control field is computed as `U_field = control_lifting_map @ U`.

Additional metadata:
- `env_id`
- `ctrl_mode`
- `dataset_modes`
- `env_kwargs`
- `random_signal_kwargs`
- `controller_class`
- `controller_source`
- `trained_in_function`
- `ppo_init_kwargs`
- `ppo_train_kwargs`
- `control_lifting_map_type`
- `control_lifting_map`
- `seeds`
- timing metrics: `traj_times_sec`, `total_time_sec`, `mean_time_per_traj_sec`

`controller_source` is `not_requested` when `ctrl` was not generated, `new` when the generator constructed the PPO controller internally, and `provided` when a controller was passed in. A controller loaded from `ppo_checkpoint_dir` by the CLI is passed into the generator, so it is recorded as `provided`.

## Default `SmoothRandom` Settings
If `random_signal_kwargs` is omitted, the generator uses:

```python
{
    "signal_mode": "manual",
    "freq_range": (1, 3),
    "min_val": -1,
    "max_val": 1,
    "amp_range": (10, 20),
}
```

`env` and `seed` are reserved for internal use and are overwritten by the generator.

The default YAML config may override these values. For example, `nonlinearPDE_data_gen.default.yaml` currently uses `signal_mode: env` with `cycles_range`, `amp_range`, `num_modes`, `zero_mean`, and `per_dim`.

## PPO Controller Settings
`ctrl` mode uses PPO only.

PPO initialization kwargs:
- `actor_hidden_dim`
- `critic_hidden_dim`
- `lr`
- `discount_factor`
- `device`

PPO training kwargs:
- `num_train_iter`
- `num_episodes_per_iter`
- `episode_length`
- `sgd_epoch_num`
- `mini_batch_size`
- `clip`
- `cov_param`

If `train_controller=True`, the generator calls `controller.train(...)` before collecting data. If a controller is provided, the generator checks that its observation and action dimensions match the rollout environment, then attaches it to the cloned `ctrl` environment.

The CLI can also load a PPO controller before generation:

```bash
.venv/bin/python nonlinearPDE_data_gen.py \
    --config nonlinearPDE_data_gen.default.yaml \
    --dataset-modes zero ctrl \
    --ppo-checkpoint-dir /path/to/ppo_run
```

## Usage
Generate aligned data for zero and smooth-random modes:

```python
data = generate_controlled_dataset_nonlinear(
    env_id="burgers",
    n_traj=300,
    dataset_modes=("zero", "random"),
    random_signal_kwargs={
        "signal_mode": "manual",
        "freq_range": (1, 3),
        "min_val": -1,
        "max_val": 1,
        "amp_range": (10, 20),
    },
    n_state=256,
    n_observation=256,
    n_action=8,
    n_steps=400,
    process_noise_cov=0,
    sensor_noise_cov=1e-8,
    action_limit=1.0,
    init_seed=123,
    noise_seed=10_000,
)
```

Generate aligned data with PPO control by training the controller inside the generator:

```python
data = generate_controlled_dataset_nonlinear(
    env_id="burgers",
    n_traj=300,
    dataset_modes=("zero", "random", "ctrl"),
    ctrl_mode="ppo",
    train_controller=True,
    ppo_train_kwargs={
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
    n_state=256,
    n_observation=256,
    n_action=8,
    n_steps=400,
    process_noise_cov=0,
    sensor_noise_cov=1e-8,
    action_limit=1.0,
    init_seed=123,
    noise_seed=10_000,
)
```

Run from the command line with the default YAML config:

```bash
.venv/bin/python nonlinearPDE_data_gen.py --config nonlinearPDE_data_gen.default.yaml
```

Run from the command line with CLI overrides:

```bash
.venv/bin/python nonlinearPDE_data_gen.py \
    --config nonlinearPDE_data_gen.default.yaml \
    --num-traj 300 \
    --dataset-modes zero random
```

Plot PDE trajectories:

```python
traj_idx = 0
_plot_single_from_data(data, traj_idx=traj_idx, mode="zero")
_plot_single_from_data(data, traj_idx=traj_idx, mode="random")
_plot_single_from_data(data, traj_idx=traj_idx, mode="ctrl")
```

Plot actuator evolution:

```python
_plot_controls_from_data(data, traj_idx=traj_idx, mode="zero")
_plot_controls_from_data(data, traj_idx=traj_idx, mode="random")
_plot_controls_from_data(data, traj_idx=traj_idx, mode="ctrl")
```

Export a single mode:

```python
ds_zero = build_export_dataset(data, mode="zero")
ds_random = build_export_dataset(data, mode="random")
ds_ctrl = build_export_dataset(data, mode="ctrl")
```

## Export Format
`build_export_dataset(...)` converts plotting-friendly arrays into time-major arrays:

```python
ds = {
    "solutions":      (N, n_steps + 1, n_state),
    "init_states":    (N, 1, n_state),
    "controls":       (N, n_steps, n_action),
    "controls_field": (N, n_steps, n_state),
    "R":              (N,),
    "T":              (N,),
    "env_id": ...,
    "ctrl_mode": ...,
    "dataset_modes": ...,
    "env_kwargs": ...,
    "random_signal_kwargs": ...,
    "seeds": ...,
    "export_mode": ...,
    "control_lifting_map_type": ...,
    "control_lifting_map": ...,
}
```

In the exported dataset:
- `R` is copied from the mode-specific cumulative reward vector.
- `T` is copied from the mode-specific valid horizon vector.
- `controls_field` is the time-major version of `U_field`, which was constructed from the mode-specific control lifting map during generation.
- `control_lifting_map_type` records whether the map came from `B2`, `control_sup`, or neither.
- `control_lifting_map` stores the actual matrix used for that export mode, or `None` when no map was available.

Supported export modes are the requested dataset modes:
- `mode="zero"`
- `mode="random"`
- `mode="ctrl"`

Optional:
- `fill_nan=True` replaces NaN padding with zeros before export

Save all requested exports from the CLI:

```python
save_export_datasets(
    data=data,
    num_traj=num_traj,
    output_dir=output_dir,
    fill_nan=fill_nan,
)
```

This writes one compressed file per requested mode:

```text
{env_id}_dataset_{mode}_{num_traj}.npz
```

The NPZ export keeps the legacy `B2` key, but for nonlinear PDEs it stores the actual control lifting matrix used by that mode. Metadata such as `dataset_modes`, `env_kwargs`, `random_signal_kwargs`, and `seeds` is stored as JSON strings.

## Sanity Checks
After generation with `dataset_modes=("zero", "random", "ctrl")`:

```python
for mode in ("zero", "random", "ctrl"):
    assert data["X"][mode].shape == (300, 256, 401)
    assert data["U"][mode].shape == (300, 8, 400)
    assert data["U_field"][mode].shape == (300, 256, 400)
    assert data["R"][mode].shape == (300,)
    assert data["T"][mode].shape == (300,)
```

Aligned initial states:

```python
for mode in data["dataset_modes"]:
    assert np.allclose(data["X"][mode][:, :, 0], data["init_states"])
```

Valid horizons can be used to crop away NaN padding:

```python
mode = "random"
k = 0
T = int(data["T"][mode][k])
X_valid = data["X"][mode][k, :, : T + 1]
U_valid = data["U"][mode][k, :, :T]
F_valid = data["U_field"][mode][k, :, :T]
```

## Reproducibility Notes
- `init_seed + k` controls initial-state sampling only.
- `noise_seed + k` controls rollout randomness and the `SmoothRandom` action trajectory for trajectory `k`.
- `zero`, `random`, and `ctrl` use the same `x0` for the same `k` when those modes are requested.
- The PPO controller is evaluated deterministically during dataset rollout with `cov_param=0.0`.
- If `env_kwargs.target_state` is a scalar, it is expanded to a full vector of shape `(n_state,)`.
- For environments that support target states but do not have one set, the generator preserves legacy behavior by setting the target state to zero.
- Control lifting maps are collected per rollout environment and stored in `data["control_lifting_map"]`; unlike the linear generator, the nonlinear generator does not require a single shared `B2` across all modes.
