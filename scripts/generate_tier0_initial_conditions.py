#!/usr/bin/env python3
"""Generate Tier 0 held-out initial conditions without solver rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-controlgym")

SCRIPT_DIR = Path(__file__).resolve().parent
CONTROLGYM_ROOT = Path("/home/pkhuran3/scratchjdrgona1/pkhuran3/projects_code/controlgym")
if CONTROLGYM_ROOT.exists():
    sys.path.insert(0, str(CONTROLGYM_ROOT))

from linearPDE_data_gen import (  # noqa: E402
    _load_yaml_config as load_linear_config,
    _make_envs as make_linear_envs,
    _sample_initial_state as sample_linear_ic,
)
from nonlinearPDE_data_gen import (  # noqa: E402
    _load_yaml_config as load_nonlinear_config,
    _make_nonlinear_envs as make_nonlinear_envs,
    _sample_initial_state as sample_nonlinear_ic,
)


DATA_ROOT = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name == "scripts"
    else Path("/home/pkhuran3/scratchjdrgona1/pkhuran3/PDEControlDPC/data")
)
DEFAULT_PDES = [
    "allen_cahn",
    "burgers",
    "cahn_hillard",
    "cdr",
    "fisher",
    "ginzburg_landau",
    "korteweg_de_vries",
    "ks_l22",
    "schrodinger",
    "wave",
]
OUTPUT_FILE = "tier0_initial_conditions.npz"


def stable_json(value) -> str:
    """Serialize to JSON with sorted keys so the same value always hashes the same."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def source_config(config: dict) -> dict:
    """Strip the run-specific output_dir so the hash reflects only generator settings."""
    return {key: value for key, value in config.items() if key != "output_dir"}


def find_config(data_root: Path, pde: str) -> tuple[Path, str]:
    """Locate a PDE's linear or nonlinear DPC config file."""
    pde_dir = data_root / pde
    paths = {
        "linear": pde_dir / "linearPDE_data_gen.dpc.yaml",
        "nonlinear": pde_dir / "nonlinearPDE_data_gen.dpc.yaml",
    }
    for kind, path in paths.items():
        if path.exists():
            return path, kind
    raise FileNotFoundError(f"No linear/nonlinear DPC config found for {pde} in {pde_dir}")


def load_config(path: Path, kind: str) -> dict:
    """Load a DPC config with the loader matching its linear/nonlinear kind."""
    return load_linear_config(path) if kind == "linear" else load_nonlinear_config(path)


def config_hash(pde: str, kind: str, config: dict, n: int, init_seed: int) -> str:
    """Hash the generator config + seed range so registry entries can detect drift."""
    payload = {
        "pde": pde,
        "generator_script": f"{kind}PDE_data_gen.py",
        "source_config": source_config(config),
        "env_id": config["env_id"],
        "env_kwargs": config["env_kwargs"],
        "N": n,
        "init_seed": init_seed,
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def control_lifting_matrix(env) -> np.ndarray:
    """Return the environment's control-to-state lifting matrix (B2 or control_sup)."""
    if hasattr(env, "B2") and getattr(env, "B2") is not None:
        return np.asarray(env.B2).copy()
    if hasattr(env, "control_sup") and getattr(env, "control_sup") is not None:
        return np.asarray(env.control_sup).copy()
    raise AttributeError("Environment is missing required B2/control_sup lifting matrix.")


def sample_ics(config: dict, kind: str, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the PDE env and sample one initial condition per seed."""
    env_id = config["env_id"]
    env_kwargs = dict(config["env_kwargs"])
    print(f"  creating environment: env_id={env_id} env_kwargs={env_kwargs}", flush=True)
    if kind == "linear":
        env_init, _ = make_linear_envs(env_id, (), **env_kwargs)
        sampler = sample_linear_ic
    else:
        env_init, _ = make_nonlinear_envs(env_id, (), **env_kwargs)
        sampler = sample_nonlinear_ic
    print("  sampling initial conditions...", flush=True)
    init_states = []
    for index, seed in enumerate(seeds, start=1):
        start = time.perf_counter()
        init_states.append(sampler(env_init, int(seed)))
        elapsed = time.perf_counter() - start
        print(f"  [{index}/{len(seeds)}] seed={int(seed)} took {elapsed:.3f}s", flush=True)
    init_states = np.stack(init_states, axis=0)
    return init_states, control_lifting_matrix(env_init)


def update_registry(path: Path, entries: list[dict]) -> None:
    """Append new registry entries, replacing any existing entry for the same PDE."""
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            registry = yaml.safe_load(fh) or {}
    else:
        registry = {"version": 1, "entries": []}

    by_pde = {
        item["pde"]: item
        for item in registry.get("entries", [])
        if isinstance(item, dict) and "pde" in item
    }
    by_pde.update({item["pde"]: item for item in entries})
    registry["entries"] = list(by_pde.values())
    registry.setdefault("version", 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(registry, fh, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen Tier 0 held-out PDE initial conditions."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-subdir", default="tier0_heldout_id")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--init-seed", type=int, default=2001)
    parser.add_argument("--pdes", nargs="+", default=DEFAULT_PDES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be positive.")

    seeds = np.arange(args.init_seed, args.init_seed + args.n, dtype=np.int64)
    if set(seeds.tolist()) & {123, 999}:
        raise ValueError("Tier 0 seeds must be disjoint from {123, 999}.")

    registry_entries = []
    data_root = args.data_root.expanduser()
    registry_path = data_root / "tier0_seed_registry.yaml"
    for pde in args.pdes:
        config_path, kind = find_config(data_root, pde)
        config = load_config(config_path, kind)
        digest = config_hash(pde, kind, config, args.n, args.init_seed)
        output_path = data_root / pde / args.output_subdir / OUTPUT_FILE

        print(f"{pde}:", flush=True)
        print(f"  config: {config_path}", flush=True)
        print(f"  output: {output_path}", flush=True)
        print(f"  env_id: {config['env_id']}", flush=True)
        print(f"  seeds: {int(seeds[0])}..{int(seeds[-1])}", flush=True)
        print(f"  generator_config_hash: {digest}", flush=True)

        if args.dry_run:
            continue
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        init_states, b2 = sample_ics(config, kind, seeds)
        save_fields = {
            "init_states": init_states,
            "seeds": seeds,
            "B2": b2,
            "pde": np.array(pde),
            "env_id": np.array(config["env_id"]),
            "N": np.array(args.n, dtype=np.int64),
            "init_seed": np.array(args.init_seed, dtype=np.int64),
            "config_path": np.array(str(config_path.resolve())),
            "generator_script": np.array(f"{kind}PDE_data_gen.py"),
            "generator_config_hash": np.array(digest),
            "env_kwargs_json": np.array(stable_json(config["env_kwargs"])),
        }
        np.savez_compressed(output_path, **save_fields)
        print("  saved fields:", flush=True)
        for name, value in save_fields.items():
            print(f"    {name}: shape={value.shape} dtype={value.dtype}", flush=True)

        registry_entries.append(
            {
                "pde": pde,
                "init_seed": args.init_seed,
                "N": args.n,
                "generator_config_hash": digest,
                "output_path": str(output_path.resolve()),
                "seed_start": int(seeds[0]),
                "seed_end": int(seeds[-1]),
            }
        )

    if args.dry_run:
        print("Dry run only; no files written.", flush=True)
    else:
        update_registry(registry_path, registry_entries)
        print(f"Wrote registry: {registry_path}", flush=True)


if __name__ == "__main__":
    main()
