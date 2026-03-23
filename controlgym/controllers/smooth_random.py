import numpy as np


SUPPORTED_PDE_ENVS = [
    "allen_cahn",
    "burgers",
    "cahn_hilliard",
    "convection_diffusion_reaction",
    "fisher",
    "ginzburg_landau",
    "korteweg_de_vries",
    "kuramoto_sivashinsky",
    "schrodinger",
    "wave",
]


class SmoothRandom:
    """
    ### Description

    This environment implements an open-loop smooth random controller for excitation
    and dataset generation. The controller precomputes a smooth bounded action
    trajectory over one episode and returns the action corresponding to the current
    environment step. The signal design is controlled by signal_mode: manual mode
    uses explicit control bounds and frequencies, while env mode uses
    environment-derived amplitude bounds when available and low-frequency cycles
    over the full rollout. An optional one-line summary is printed when a
    trajectory is generated.

    ### Arguments
    For env_id in the following list:
    ["allen_cahn", "burgers", "cahn_hilliard",
    "convection_diffusion_reaction", "fisher", "ginzburg_landau",
    "korteweg_de_vries", "kuramoto_sivashinsky", "schrodinger", "wave"]

    ```
    env = controlgym.make("allen_cahn", n_steps=200, n_action=8)
    controller = controlgym.controllers.SmoothRandom(
        env,
        seed=123,
        signal_mode="env",
        cycles_range=(1.0, 3.0),
        num_modes=4,
        verbose=True,
    )
    total_reward = controller.run(seed=0)

    controller = controlgym.controllers.SmoothRandom(
        env,
        seed=123,
        signal_mode="manual",
        freq_range=(1, 3),
        min_val=-1,
        max_val=1,
        amp_range=(10, 20),
    )
    total_reward = controller.run(seed=0)
    ```

    Arguments:
        env: object, controlgym environment object.
        seed: int, random seed for reproducible action trajectories.
        min_val: float or ndarray[float], minimum control value used in signal_mode="manual".
        max_val: float or ndarray[float], maximum control value used in signal_mode="manual".
        signal_mode: str, "manual" uses min_val, max_val, and freq_range; "env" uses
            environment-derived amplitude bounds and cycles_range.
        num_modes: int, number of sinusoidal modes in the random signal.
        freq_range: tuple[float, float], explicit frequency range used in signal_mode="manual".
        cycles_range: tuple[float, float], rollout-scale cycle range used in signal_mode="env".
        amp_range: tuple[float, float], range of amplitudes of each mode.
        zero_mean: bool, whether to subtract the mean before scaling.
        per_dim: bool, whether to generate independent signals for each action dimension.
        verbose: bool, whether to print a short summary when a trajectory is generated.
    """

    def __init__(
        self,
        env,
        seed: int = None,
        min_val: float = -1.0,
        max_val: float = 1.0,
        num_modes: int = 4,
        freq_range: tuple[float, float] = (1.0, 3.0),
        amp_range: tuple[float, float] = (1.0, 3.0),
        zero_mean: bool = True,
        per_dim: bool = True,
        signal_mode: str = "manual",
        cycles_range: tuple[float, float] = None,
        verbose: bool = False,
    ):
        self.env = env
        self.seed = seed
        self.signal_mode = signal_mode
        self.num_modes = num_modes
        self.freq_range = freq_range
        self.cycles_range = freq_range if cycles_range is None else cycles_range
        self.amp_range = amp_range
        self.zero_mean = zero_mean
        self.per_dim = per_dim
        self.verbose = verbose

        self.action_shape = self.env.action_space.shape
        self.action_dtype = np.dtype(self.env.action_space.dtype)
        self.n_action = int(np.prod(self.action_shape))

        self.min_val = self._broadcast_bound(min_val)
        self.max_val = self._broadcast_bound(max_val)

        assert getattr(self.env, "category", None) == "pde" and getattr(
            self.env, "id", None
        ) in SUPPORTED_PDE_ENVS, (
            "SmoothRandom only supports PDE environments: "
            + str(SUPPORTED_PDE_ENVS)
        )
        assert self.n_action > 0, "The action dimension must be positive"
        assert self.env.n_steps > 0, "The episode length must be positive"
        assert self.signal_mode in [
            "manual",
            "env",
        ], 'signal_mode must be either "manual" or "env"'
        assert self.num_modes > 0, "num_modes must be positive"
        assert len(self.freq_range) == 2 and self.freq_range[0] <= self.freq_range[1], (
            "freq_range must be a tuple of two ordered values"
        )
        assert len(self.cycles_range) == 2 and self.cycles_range[0] <= self.cycles_range[1], (
            "cycles_range must be a tuple of two ordered values"
        )
        assert len(self.amp_range) == 2 and self.amp_range[0] <= self.amp_range[1], (
            "amp_range must be a tuple of two ordered values"
        )
        assert np.all(
            self.min_val <= self.max_val
        ), "min_val must be less than or equal to max_val"

        self.rng = np.random.default_rng(seed=seed)
        self.action_traj = None
        self.reset()

    def _broadcast_bound(self, value):
        """Private function to broadcast scalar or array bounds to the action shape."""
        return np.broadcast_to(np.asarray(value, dtype=float), self.action_shape).copy()

    def _get_rollout_dt(self):
        """Private function to infer the rollout time step from the environment."""
        return float(self.env.sample_time)

    def _resolve_amplitude_bounds(self):
        """Private function to resolve the final control amplitude bounds."""
        if self.signal_mode == "manual":
            return self.min_val, self.max_val, "manual_bounds"

        action_limit = getattr(self.env, "action_limit", None)
        if action_limit is not None and np.isfinite(action_limit):
            bound = self._broadcast_bound(abs(action_limit))
            return -bound, bound, "env.action_limit"

        if hasattr(self.env, "init_amplitude_mean") or hasattr(
            self.env, "init_amplitude_width"
        ):
            amplitude_mean = abs(float(getattr(self.env, "init_amplitude_mean", 0.0)))
            amplitude_width = abs(float(getattr(self.env, "init_amplitude_width", 0.0)))
            amplitude_scale = amplitude_mean + 0.5 * amplitude_width
            if amplitude_scale > 0:
                bound = self._broadcast_bound(amplitude_scale)
                return -bound, bound, "env.init_amplitude"

        return self.min_val, self.max_val, "manual_fallback"

    def _generate_signal(self, time: np.ndarray[float], rollout_time: float):
        """Private function to generate a smooth random signal over one episode."""
        signal = np.zeros(time.shape[0], dtype=float)
        signal_param_used = np.zeros(self.num_modes, dtype=float)
        for mode_idx in range(self.num_modes):
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            amp = self.rng.uniform(self.amp_range[0], self.amp_range[1])
            if self.signal_mode == "manual":
                freq = self.rng.uniform(self.freq_range[0], self.freq_range[1])
                signal += amp * np.sin(2.0 * np.pi * freq * time + phase)
                signal_param_used[mode_idx] = freq
            else:
                cycles = self.rng.uniform(self.cycles_range[0], self.cycles_range[1])
                signal += amp * np.sin(2.0 * np.pi * cycles * time / rollout_time + phase)
                signal_param_used[mode_idx] = cycles

        if self.zero_mean:
            signal -= np.mean(signal)

        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal /= max_abs

        return signal, signal_param_used

    def _print_generation_summary(
        self,
        min_val: np.ndarray[float],
        max_val: np.ndarray[float],
        amplitude_source: str,
        signal_param_used: np.ndarray[float],
        action_traj: np.ndarray[float],
        dt: float,
    ):
        """Private function to print a short summary of the generated trajectory."""
        if not self.verbose:
            return

        dt_str = "n/a" if dt is None else str(dt)
        signal_param_name = "freq_used" if self.signal_mode == "manual" else "cycles_used"
        print(
            "[SmoothRandom] "
            + f"action_dim={self.n_action}, "
            + f"T={self.env.n_steps}, "
            + f"dt={dt_str}, "
            + f"signal_mode={self.signal_mode}, "
            + f"amplitude_source={amplitude_source}, "
            + f"final_range=({min_val.min():.3f}, {max_val.max():.3f}), "
            + f"{signal_param_name}=({signal_param_used.min():.3f}, {signal_param_used.max():.3f}), "
            + f"signal_min={action_traj.min():.3f}, "
            + f"signal_max={action_traj.max():.3f}"
        )

    def _generate_action_traj(self):
        """Private function to precompute the open-loop action trajectory."""
        dt = self._get_rollout_dt()
        rollout_time = 1.0 if dt is None else self.env.n_steps * dt
        if rollout_time <= 0:
            rollout_time = 1.0

        time = np.linspace(0.0, rollout_time, self.env.n_steps)
        signal_traj = np.zeros((self.env.n_steps, self.n_action), dtype=float)
        signal_param_used = []
        min_val, max_val, amplitude_source = self._resolve_amplitude_bounds()

        if self.per_dim:
            for action_idx in range(self.n_action):
                signal_traj[:, action_idx], signal_params = self._generate_signal(
                    time, rollout_time
                )
                signal_param_used.append(signal_params)
        else:
            shared_signal, signal_params = self._generate_signal(time, rollout_time)
            signal_traj = np.repeat(shared_signal[:, np.newaxis], self.n_action, axis=1)
            signal_param_used.append(signal_params)

        signal_traj = signal_traj.reshape((self.env.n_steps,) + self.action_shape)
        action_center = (max_val + min_val) / 2.0
        action_radius = (max_val - min_val) / 2.0
        action_traj = signal_traj * action_radius[np.newaxis, ...] + action_center[np.newaxis, ...]
        action_traj = np.clip(action_traj, min_val, max_val)
        action_traj = action_traj.astype(self.action_dtype, copy=False)

        self._print_generation_summary(
            min_val=min_val,
            max_val=max_val,
            amplitude_source=amplitude_source,
            signal_param_used=np.concatenate(signal_param_used),
            action_traj=action_traj,
            dt=dt,
        )
        return action_traj

    def reset(self, seed: int = None):
        """Regenerate the open-loop action trajectory."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed=seed)
        self.action_traj = self._generate_action_traj()

    def select_action(self):
        """Returns the precomputed control input for the current step."""
        step = min(self.env.step_count, self.env.n_steps - 1)
        return self.action_traj[step].copy()

    def run(self, state=None, seed=None):
        """Run a trajectory of the environment using smooth random control,
            calculate the H2 cost, and save the state trajectory to env.state_traj.
            The trajectory is terminated when the environment returns a done signal (most likely
            due to the exceedance of the maximum number of steps: env.n_steps)
        Args:
            state: (optional ndarray[float]), an user-defined initial state.
            seed: (optional int), random seed for the environment.

        Returns:
            total_reward: float, the accumulated reward of the trajectory,
                which is equal to the negative H2 cost.
        """
        # reset the environment
        _, info = self.env.reset(seed=seed, state=state)
        # run the simulated trajectory and calculate the h2 cost
        total_reward = 0
        state_traj = np.zeros((self.env.n_state, self.env.n_steps + 1))
        state_traj[:, 0] = info["state"]
        for t in range(self.env.n_steps):
            action = self.select_action()
            observation, reward, terminated, truncated, info = self.env.step(action)

            state_traj[:, t + 1] = info["state"]
            if terminated or truncated:
                break
            total_reward += reward

        self.env.state_traj = state_traj
        return total_reward
