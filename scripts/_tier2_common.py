"""Shared helpers for the Tier 2 noisy-IC generators (burn-in and additive).

Loading conventions for the frozen Tier 0 npz are reused from
generate_tier1_amplitude_initial_conditions.py (scalar_value/json_value/
control_lifting_matrix/stable_json), which already reads this exact file
format; nothing here re-implements that parsing.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import generate_tier1_amplitude_initial_conditions as _tier1  # noqa: E402

stable_json = _tier1.stable_json
DATA_ROOT = _tier1.DATA_ROOT

K_BURNIN = 10
NOISE_SEED_BASE = 70000
OUTPUT_FILE_NAME = "tier2_initial_conditions.npz"
REGISTRY_PATH = _SCRIPTS_DIR.parent / "configs" / "tier2_registry.yaml"

# Locked levels (Prompt 2). Insertion order is the canonical processing order.
DELTA_TARGETS = {"0": 0.0, "1e-2": 1e-2, "2e-1": 2e-1}
BURNIN_LEVELS = tuple(DELTA_TARGETS.keys())
ADDITIVE_LEVELS = tuple(tag for tag in DELTA_TARGETS if tag != "0")


def discover_pdes(data_root: Path) -> list[str]:
    """Find every PDE with a frozen Tier-0 npz under data_root (mirrors tier1's --all discovery)."""
    data_root = Path(data_root)
    paths = sorted(data_root.glob("*/tier0_heldout_id/tier0_initial_conditions.npz"))
    if not paths:
        raise FileNotFoundError(f"No Tier-0 held-out IC files found under {data_root}")
    return [p.parents[1].name for p in paths]


def load_tier0(data_root: Path, pde: str) -> dict:
    """Load one PDE's frozen Tier-0 npz, validated against the structure written by
    generate_tier0_initial_conditions.py. Raises on any structural mismatch instead of
    adapting silently (per hard constraint)."""
    path = Path(data_root) / pde / "tier0_heldout_id" / "tier0_initial_conditions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Tier-0 IC file not found for pde={pde!r}: {path}")

    with np.load(path, allow_pickle=True) as data:
        required = {
            "init_states", "seeds", "N", "pde", "env_id",
            "init_seed", "env_kwargs_json", "generator_config_hash",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"Tier-0 npz at {path} is missing expected key(s) {sorted(missing)}; "
                "structure does not match generate_tier0_initial_conditions.py. Stopping "
                "rather than adapting silently."
            )

        init_states = np.asarray(data["init_states"], dtype=float)
        if init_states.ndim != 2:
            raise ValueError(
                f"Tier-0 npz at {path} has init_states.ndim={init_states.ndim}; expected 2 "
                "(N, n_state), as written by generate_tier0_initial_conditions.py's "
                "np.stack(init_states, axis=0). Stopping rather than adapting silently."
            )

        n_val = int(_tier1.scalar_value(data, "N"))
        if init_states.shape[0] != n_val:
            raise ValueError(
                f"Tier-0 npz at {path}: init_states.shape[0]={init_states.shape[0]} "
                f"!= N={n_val}. Stopping rather than adapting silently."
            )

        env_kwargs = _tier1.json_value(data, "env_kwargs_json")
        if not isinstance(env_kwargs, dict) or "n_state" not in env_kwargs:
            raise ValueError(
                f"Tier-0 npz at {path}: env_kwargs_json does not decode to a dict "
                "containing 'n_state'. Stopping rather than adapting silently."
            )
        if int(env_kwargs["n_state"]) != init_states.shape[1]:
            raise ValueError(
                f"Tier-0 npz at {path}: env_kwargs.n_state={env_kwargs['n_state']} != "
                f"init_states.shape[1]={init_states.shape[1]}. Stopping rather than "
                "adapting silently."
            )

        pde_field = _tier1.scalar_value(data, "pde")
        if pde_field != pde:
            raise ValueError(
                f"Tier-0 npz at {path}: pde field {pde_field!r} != requested {pde!r}. "
                "Stopping rather than adapting silently."
            )

        _, lifting_matrix = _tier1.control_lifting_matrix(data)

        return {
            "path": path,
            "pde": pde_field,
            "env_id": _tier1.scalar_value(data, "env_id"),
            "env_kwargs": env_kwargs,
            "init_states": init_states,
            "seeds": np.asarray(data["seeds"]).copy(),
            "B2": lifting_matrix,
            "N": n_val,
            "init_seed": int(_tier1.scalar_value(data, "init_seed")),
            "generator_config_hash": _tier1.scalar_value(data, "generator_config_hash"),
        }


def compute_med_norm(init_states: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(init_states, axis=1)))


def delta_stats(deltas: np.ndarray) -> tuple[float, float]:
    deltas = np.asarray(deltas, dtype=float)
    median = float(np.median(deltas))
    iqr = float(np.percentile(deltas, 75) - np.percentile(deltas, 25))
    return median, iqr


def savez_deterministic(path: Path, **fields) -> None:
    """np.savez_compressed equivalent, but with fixed zip member timestamps so that two
    runs with identical array content produce byte-identical files (np.savez_compressed
    stamps each zip member with the current wall-clock time, which breaks --verify)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    members = {}
    for name, value in fields.items():
        arr = np.asarray(value)
        buf = io.BytesIO()
        np.lib.format.write_array(buf, arr, allow_pickle=False)
        members[f"{name}.npy"] = buf.getvalue()

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname in sorted(members):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, members[arcname])


def append_registry_entries(entries: list[dict]) -> None:
    """Append/replace configs/tier2_registry.yaml entries keyed by (pde, option, level_tag)."""
    path = REGISTRY_PATH
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            registry = yaml.safe_load(fh) or {}
    else:
        registry = {"version": 1, "entries": []}

    def key(item):
        return (item["pde"], item["option"], item["level_tag"])

    by_key = {key(item): item for item in registry.get("entries", []) if isinstance(item, dict)}
    by_key.update({key(entry): entry for entry in entries})
    registry["entries"] = [by_key[k] for k in sorted(by_key)]
    registry.setdefault("version", 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(registry, fh, sort_keys=False)
