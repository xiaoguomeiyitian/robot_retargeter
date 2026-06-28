#!/usr/bin/env python3
"""Config loader with inheritance support.

Allows robot configs to inherit from a base config using the 'extends' key.
When a config has 'extends: base_name', it loads the base config first,
then overlays the child's fields on top.

Usage:
    from config_loader import load_robot_config
    config = load_robot_config("config/robot/h2.yaml")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "robot"
INHERITANCE_MAX_DEPTH = 5


def load_robot_config(config_path: str | Path) -> dict[str, Any]:
    """Load a robot config, resolving inheritance if 'extends' is present.

    Args:
        config_path: Path to the YAML config file
    Returns:
        Fully resolved config dict
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = CONFIG_DIR / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    # Resolve inheritance
    _resolve_inheritance(config, config_path.parent, depth=0)
    return config


def _resolve_inheritance(
    config: dict[str, Any],
    config_dir: Path,
    depth: int,
) -> None:
    """Recursively resolve 'extends' in config (in-place)."""
    if depth >= INHERITANCE_MAX_DEPTH:
        raise RecursionError(
            f"Config inheritance depth exceeds {INHERITANCE_MAX_DEPTH}. "
            "Possible circular dependency."
        )

    extends = config.get("extends")
    if not extends:
        return

    # Load base config
    base_path = config_dir / f"{extends}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base config '{extends}' not found at {base_path}. "
            f"Referenced by config with extends: {extends}"
        )

    with open(base_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Recursively resolve base's inheritance
    _resolve_inheritance(base_config, config_dir, depth + depth + 1)

    # Overlay: child fields take precedence
    # For dict fields, merge recursively
    # For list fields, child replaces base
    merged = _deep_merge(base_config, config)

    # Clear and update config in-place
    config.clear()
    config.update(merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override takes precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue  # Don't carry extends into merged result
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_inheritance_chain(config_path: str | Path) -> list[str]:
    """Get the inheritance chain for a config (for debugging)."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = CONFIG_DIR / config_path

    chain = []
    current = config_path
    visited = set()

    while current is not None and current.name not in visited:
        visited.add(current.name)
        chain.append(current.stem)

        with open(current, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        extends = config.get("extends")
        if not extends:
            break

        current = current.parent / f"{extends}.yaml"
        if not current.exists():
            chain.append(f"⚠️  {extends} (NOT FOUND)")
            break

    return chain


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python config_loader.py <config_name> [--chain]")
        print("Example: python config_loader.py g1 --chain")
        sys.exit(1)

    name = sys.argv[1]
    show_chain = "--chain" in sys.argv

    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    if show_chain:
        chain = get_inheritance_chain(config_path)
        print(f"Inheritance chain: {' → '.join(chain)}")
    else:
        config = load_robot_config(config_path)
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
