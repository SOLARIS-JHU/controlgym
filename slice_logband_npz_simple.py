#!/usr/bin/env python3
"""Slice a ControlGym log-band NPZ dataset along the time axis.

This script keeps the first dimension, trajectory count, unchanged.
It removes the first 2500 control steps and keeps the next 200 control steps.

Expected input shapes:
    solutions:      (N, 3001, n_state)
    controls:       (N, 3000, n_action)
    controls_field: (N, 3000, n_state)

Output shapes:
    solutions:      (N, 201, n_state)
    init_states:    (N, 1, n_state)
    controls:       (N, 200, n_action)
    controls_field: (N, 200, n_state)

All other NPZ keys are copied unchanged by default.
"""

from pathlib import Path

import numpy as np


INPUT_NPZ = Path(
    "/scratch/jdrgona1/pkhuran3/PDEControlDPC/data/ks_l22/dpc/kuramoto_sivashinsky_dataset_zero_500.npz"
)
OUTPUT_NPZ = Path(
    "/scratch/jdrgona1/pkhuran3/PDEControlDPC/data/ks_l22/dpc/kuramoto_sivashinsky_dataset_zero_500_skip2500_next200.npz"
)

SKIP_STEPS = 2500
KEEP_STEPS = 200

# Your request says to maintain all other fields unchanged.
# Leave this False to keep T exactly as it is in the source NPZ.
# Set this True if you want T to describe the sliced 200-step horizon.
UPDATE_T_TO_SLICED_HORIZON = False


print(f"Reading: {INPUT_NPZ}")
print(f"Writing: {OUTPUT_NPZ}")
print(f"Skipping first {SKIP_STEPS} steps, keeping next {KEEP_STEPS} steps")

with np.load(INPUT_NPZ, allow_pickle=True) as data:
    print("\nInput shapes:")
    print("  solutions:     ", data["solutions"].shape)
    print("  init_states:   ", data["init_states"].shape)
    print("  controls:      ", data["controls"].shape)
    print("  controls_field:", data["controls_field"].shape)
    print("  T:             ", data["T"].shape, "min=", data["T"].min(), "max=", data["T"].max())

    state_start = SKIP_STEPS
    state_stop = SKIP_STEPS + KEEP_STEPS + 1
    control_start = SKIP_STEPS
    control_stop = SKIP_STEPS + KEEP_STEPS

    if state_stop > data["solutions"].shape[1]:
        raise ValueError(
            f"solutions has only {data['solutions'].shape[1]} time samples, "
            f"but requested stop index is {state_stop}."
        )
    if control_stop > data["controls"].shape[1]:
        raise ValueError(
            f"controls has only {data['controls'].shape[1]} time samples, "
            f"but requested stop index is {control_stop}."
        )
    if control_stop > data["controls_field"].shape[1]:
        raise ValueError(
            f"controls_field has only {data['controls_field'].shape[1]} time samples, "
            f"but requested stop index is {control_stop}."
        )

    # Slice the large time-series arrays.
    # .copy() makes the output arrays independent of the original full arrays.
    sliced_solutions = data["solutions"][:, state_start:state_stop, :].copy()
    sliced_controls = data["controls"][:, control_start:control_stop, :].copy()
    sliced_controls_field = data["controls_field"][:, control_start:control_stop, :].copy()

    # The new initial state is the first state of the sliced solution window.
    sliced_init_states = sliced_solutions[:, 0:1, :].copy()

    output = {}

    # Copy every key except the ones we are replacing.
    for key in data.files:
        if key in {"solutions", "init_states", "controls", "controls_field"}:
            continue
        output[key] = data[key]

    output["solutions"] = sliced_solutions
    output["init_states"] = sliced_init_states
    output["controls"] = sliced_controls
    output["controls_field"] = sliced_controls_field

    if UPDATE_T_TO_SLICED_HORIZON:
        output["T"] = np.full_like(data["T"], KEEP_STEPS)

    print("\nOutput shapes:")
    print("  solutions:     ", output["solutions"].shape)
    print("  init_states:   ", output["init_states"].shape)
    print("  controls:      ", output["controls"].shape)
    print("  controls_field:", output["controls_field"].shape)
    print("  T:             ", output["T"].shape, "min=", output["T"].min(), "max=", output["T"].max())

    print("\nSaving compressed NPZ...")
    np.savez_compressed(OUTPUT_NPZ, **output)

print("\nDone.")
