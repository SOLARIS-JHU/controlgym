# Linear PDE Dataset Pipeline

## Scope
Supported linear PDE environments:
- `convection_diffusion_reaction`
- `wave`
- `schrodinger`

Closed-loop controller choices:
- `lqg`
- `lqr`

Dataset rollout modes:
- `zero`
- `random`
- `ctrl`

`ctrl` is the generic dataset mode name for the closed-loop controller. The actual controller type is still stored in `ctrl_mode`.

## Generator Overview
`generate_controlled_dataset(...)` now creates three aligned trajectories for every sampled initial state:
- `zero`: `gym.controllers.Zero`
- `random`: `gym.controllers.SmoothRandom`
- `ctrl`: `gym.controllers.LQG` or `gym.controllers.LQR`

For trajectory index `k`:
- initial state seed is `init_seed + k`
- rollout seed is `noise_seed + k`
- all three modes start from the same `x0`
- the `random` controller regenerates its open-loop action sequence with `random_controller.reset(seed=noise_seed + k)`

This guarantees:
- `N` sampled initial states
- `N` zero-control rollouts
- `N` smooth-random rollouts
- `N` LQG/LQR rollouts
- the same trajectory index `k` is aligned across all three modes

## Dataset Structure
The returned dictionary stores plotting-friendly arrays:

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
- `B2`: actuator-to-state input matrix shared by all three rollout environments. It maps actuator amplitudes into state-space forcing, so the stored control field is computed as `U_field = B2 @ U`.

Additional metadata:
- `env_id`
- `ctrl_mode`
- `dataset_modes`
- `env_kwargs`
- `random_signal_kwargs`
- `B2`
- timing metrics: `traj_times_sec`, `total_time_sec`, `mean_time_per_traj_sec`

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

## Usage
Generate aligned data for all three modes:

```python
data = generate_controlled_dataset(
    env_id="wave",
    n_traj=300,
    ctrl_mode="lqg",  # or "lqr"
    random_signal_kwargs={
        "signal_mode": "manual",
        "freq_range": (1, 3),
        "min_val": -1,
        "max_val": 1,
        "amp_range": (10, 20),
    },
    n_state=200,
    n_observation=200,
    n_action=8,
    n_steps=200,
    process_noise_cov=0,
    sensor_noise_cov=1e-8,
    init_seed=123,
    noise_seed=10_000,
)
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
    "controls":       (N, n_steps, n_action),
    "controls_field": (N, n_steps, n_state),
    "R":              (N,),
    "T":              (N,),
    "env_id": ...,
    "ctrl_mode": ...,
    "env_kwargs": ...,
    "random_signal_kwargs": ...,
    "export_mode": ...,
}
```

In the exported dataset:
- `R` is copied from the mode-specific cumulative reward vector.
- `T` is copied from the mode-specific valid horizon vector.
- `controls_field` is the time-major version of `U_field`, which was constructed from the shared `B2` matrix during generation.

Supported export modes:
- `mode="zero"`
- `mode="random"`
- `mode="ctrl"`

Optional:
- `fill_nan=True` replaces NaN padding with zeros before export

Save all three exports:

```python
ds_zero = build_export_dataset(data, mode="zero")
np.savez_compressed(f"{env_id}_dataset_zero_{num_traj}.npz", ...)

ds_random = build_export_dataset(data, mode="random")
np.savez_compressed(f"{env_id}_dataset_random_{num_traj}.npz", ...)

ds_ctrl = build_export_dataset(data, mode="ctrl")
np.savez_compressed(f"{env_id}_dataset_ctrl_{num_traj}.npz", ...)
```

## Sanity Checks
After generation:

```python
for mode in ("zero", "random", "ctrl"):
    assert data["X"][mode].shape == (300, 200, 201)
    assert data["U"][mode].shape == (300, 8, 200)
    assert data["U_field"][mode].shape == (300, 200, 200)
    assert data["R"][mode].shape == (300,)
    assert data["T"][mode].shape == (300,)
```

Aligned initial states:

```python
for mode in ("zero", "random", "ctrl"):
    assert np.allclose(data["X"][mode][:, :, 0], data["init_states"])
```

## Reproducibility Notes
- `init_seed + k` controls initial-state sampling only.
- `noise_seed + k` controls rollout randomness and the `SmoothRandom` action trajectory for trajectory `k`.
- `zero`, `random`, and `ctrl` use the same `x0` for the same `k`.
- The shared actuator matrix `B2` is validated across rollout environments once and stored as `data["B2"]`.
